import argparse
import os
import json
import random
import torch
import pandas as pd
import evaluate
from datasets import load_dataset

from src.eval.utils import (
    generate_completions,
    load_hooked_lm_and_tokenizer,
    dynamic_import_function
)
from src.eval.templates import create_prompt_with_llama3_chat_format
from src.utils import seed_torch

# Metrics
rouge = evaluate.load("rouge")

# Hook function for dynamic patching
def layer_patch_hook(value, hook, neurons, patched_values):
    try:
        if not isinstance(patched_values, torch.Tensor):
            patched_values = torch.tensor(patched_values)
        patched_values = patched_values.to(value)
        value[..., neurons] = patched_values
    except Exception as e:
        print(f"Error in hook {hook}: {e}")
    return value

# Simple prompt template
def generation_prompt_template(input_):
    return f"### Input:\n{input_}\n\n### Response:\n"


def main(args):
    # 1. Reproducibility
    seed_torch(42)
    random.seed(42)

    # 2. Load dataset
    ds = load_dataset('csv', data_files={'test': args.dataset}, split='test')
    if args.max_num_examples:
        ds = ds.shuffle(seed=42).select(range(args.max_num_examples))
    questions = [row['question'].strip() for row in ds]
    references = [row['answer'].strip() for row in ds]

    # 3. Build prompts
    chat_fn = None
    if args.use_chat_format:
        chat_fn = dynamic_import_function(args.chat_formatting_function)
    prompts = []
    for q in questions:
        if chat_fn:
            messages = [{"role": "user", "content": q}]
            prompt = chat_fn(messages)
        else:
            prompt = generation_prompt_template(q)
        prompts.append(prompt)

    # 4. Load models
    red_model, tokenizer = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.red_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map="auto",
        use_fast_tokenizer=not args.use_slow_tokenizer
    )
    blue_model, _ = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.blue_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map="auto",
        use_fast_tokenizer=not args.use_slow_tokenizer
    )
    base_model, _ = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.blue_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map="auto",
        use_fast_tokenizer=not args.use_slow_tokenizer
    )

    # 5. Load or generate neuron indices
    if args.random_index:
        n_layers = 15 #red_model.config.num_hidden_layers
        hidden_size = 8192 #red_model.config.hidden_size
        first_dims = torch.randint(low=0, high=n_layers, size=(args.patch_num,))
        second_dims = torch.randint(low=0, high=hidden_size, size=(args.patch_num,))
        index_tensor = torch.stack([first_dims, second_dims], dim=1)
    else:
        _, idx, *_ = torch.load(args.index_path)
        if args.patch_start is not None:
            index_tensor = idx[args.patch_start : args.patch_start + args.patch_num]
        else:
            index_tensor = idx[: args.patch_num]

    # 6. Generate outputs
    print("Generating from original (unpatched) model...")
    outputs_before = generate_completions(
        model=red_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        stop_id_sequences=[[tokenizer.eos_token_id]],
        do_sample=False,
        guided_model=None,
        index=None,
        hook_fn=layer_patch_hook
    )

    print("Generating from patched model...")
    outputs_after = generate_completions(
        model=red_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        stop_id_sequences=[[tokenizer.eos_token_id]],
        do_sample=False,
        guided_model=blue_model,
        index=index_tensor,
        hook_fn=layer_patch_hook
    )

    print("Generating from base (clean) model...")
    outputs_base = generate_completions(
        model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        stop_id_sequences=[[tokenizer.eos_token_id]],
        do_sample=False,
        guided_model=None,
        index=None,
        hook_fn=None
    )

    # 7. Evaluate ROUGE
    rouge_before = rouge.compute(predictions=outputs_before, references=references)["rougeL"]
    rouge_after = rouge.compute(predictions=outputs_after, references=references)["rougeL"]
    rouge_base = rouge.compute(predictions=outputs_base, references=references)["rougeL"]

    print(f"\nROUGE-L before patch: {rouge_before:.4f}")
    print(f"ROUGE-L after patch:  {rouge_after:.4f}")
    print(f"ROUGE-L base model:   {rouge_base:.4f}")

    # 8. Save to CSV
    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame({
        'question': questions,
        'reference': references,
        'output_before_patch': outputs_before,
        'output_after_patch': outputs_after,
        'output_base_model': outputs_base,
    })
    df.to_csv(os.path.join(args.save_dir, 'predictions.csv'), index=False)

    summary = {
        'rougeL_before_patch': rouge_before,
        'rougeL_after_patch': rouge_after,
        'rougeL_base_model': rouge_base
    }
    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as fout:
        json.dump(summary, fout, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='CSV file path')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--max_num_examples', type=int, default=None)
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True)
    parser.add_argument('--red_model_name_or_path', type=str, required=True)
    parser.add_argument('--blue_model_name_or_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, default=None)
    parser.add_argument('--patch_num', type=int, default=3000)
    parser.add_argument('--patch_start', type=int, default=None)
    parser.add_argument('--random_index', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--use_slow_tokenizer', action='store_true')
    parser.add_argument('--use_chat_format', action='store_true')
    parser.add_argument('--chat_formatting_function', type=str,
                        default='src.eval.templates.create_prompt_with_llama3_chat_format')
    args = parser.parse_args()
    main(args)
    
'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python final_scores.py \
  --dataset /data/TOFU_forget10_train.csv \
  --save_dir results/tofu10_eval \
  --tokenizer_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --red_model_name_or_path /nas/home/mhlee/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent \
  --blue_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --index_path /nas/home/mhlee/te/llama3.2-1B.pt \
  --patch_num 5000 \
  --eval_batch_size 10
  --use_chat_format \
  --use_slow_tokenizer
'''
# 깃허브 코드대로 일단 레이어 마지막 빼고 15까지함

# Intermediate size: 8192
# Layer 15
# red_name_or_path /nas/home/mhlee/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90
# blue_model_name_or_path meta-llama/Llama-3.2-1B-Instruct
# index_path: /nas/home/mhlee/te/llama3.2-1B.pt
# use_chat_format
# use_slow_tokenizer