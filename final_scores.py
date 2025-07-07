import argparse
import os
import json
import random
import torch
import pandas as pd
import evaluate
from datasets import load_dataset
import re
from tqdm import tqdm

from src.eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    load_hooked_lm_and_tokenizer,
    dynamic_import_function
)
from src.eval.templates import create_prompt_with_llama3_chat_format
from src.utils import seed_torch

# Metrics
rouge_metric = evaluate.load("rouge")

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

def generation_prompt_template(input_):
    return f"### Input:\n{input_}\n\n### Response:\n"

def main(args):
    seed_torch(42)

    ds = load_dataset('csv', data_files={'test': args.dataset}, split='test')
    if args.max_num_examples:
        ds = ds.shuffle(seed=42).select(range(args.max_num_examples))
    questions = [row['question'].strip() for row in ds]
    references = [row['answer'].strip() for row in ds]

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

    red_model, tokenizer = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.red_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map="auto",
        use_fast_tokenizer=not args.use_slow_tokenizer
    )
    guided_model, _ = load_hf_lm_and_tokenizer(
        model_name_or_path=args.blue_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map="auto",
        use_fast_tokenizer=not args.use_slow_tokenizer
    )

    n_layers = 15
    hidden_size = 8192
    causality_scores = torch.zeros(n_layers, hidden_size)

    not_patching_outputs = generate_completions(
        model=red_model,
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
    not_patching_rouge = rouge_metric.compute(predictions=not_patching_outputs, references=references)["rougeL"]

    records = []

    for i in range(n_layers):
        for j in range(0, hidden_size, 512):
            topk_index = torch.tensor([[i, j + k] for k in range(512)])

            outputs = generate_completions(
                model=red_model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[[tokenizer.eos_token_id]],
                do_sample=False,
                guided_model=guided_model,
                index=topk_index,
                hook_fn=layer_patch_hook
            )

            patched_rouge = rouge_metric.compute(predictions=outputs, references=references)["rougeL"]
            for k in range(512):
                causality_scores[i, j + k] = not_patching_rouge - patched_rouge

            for q, r, np_out, p_out in zip(questions, references, not_patching_outputs, outputs):
                rouge_np = rouge_metric.compute(predictions=[np_out], references=[r])["rougeL"]
                rouge_p = rouge_metric.compute(predictions=[p_out], references=[r])["rougeL"]
                records.append({
                    "question": q,
                    "reference": r,
                    "output_before_patch": np_out,
                    "output_after_patch": p_out,
                    "rouge_before": rouge_np,
                    "rouge_after": rouge_p
                })

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(causality_scores, os.path.join(args.save_dir, "causality_scores.pt"))
    pd.DataFrame(records).to_csv(os.path.join(args.save_dir, "rouge_patch_comparison.csv"), index=False)

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
    parser.add_argument('--chat_formatting_function', type=str, default='src.eval.templates.create_prompt_with_llama3_chat_format')
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