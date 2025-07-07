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
    load_hooked_lm_and_tokenizer,
    dynamic_import_function
)
from src.eval.templates import create_prompt_with_llama3_chat_format
from src.utils import seed_torch,topk_index
from src.activation_processor_SFT import ActivationContrasting

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

def compute_causal_score(
    a_con, a_con_patch, a_un, a_un_patch
):
    """
    C_N = (a_con - a_con_patch) + (1 - (a_un - a_un_patch))
    """
    return (a_con - a_con_patch) + (1.0 - (a_un - a_un_patch))

def main(args):
    seed_torch(42)

    # 1. Load & preprocess data
    dataset = load_dataset('csv', data_files=args.dataset)["train"]
    test_data = []
    for row in dataset:
        try:
            question = row["question"].strip()
            answer = row["answer"].strip()
            test_data.append({"question": question, "answer": answer})
        except Exception:
            continue

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    # 2. Build prompts
    prompts = []
    for ex in test_data:
        if args.use_chat_format:
            chat_fn = dynamic_import_function(args.chat_formatting_function)
            messages = [{"role": "user", "content": ex["question"]}]
            prompt = chat_fn(messages) if chat_fn else generation_prompt_template(ex["question"])
        else:
            prompt = generation_prompt_template(ex["question"])
        prompts.append(prompt)

    # 3. Activation Contrasting
    names_filter = lambda name: name.startswith('model.layers.') and name.endswith('mlp.hook_post') and '.16.' not in name
    ac = ActivationContrasting(
        args.base_model_name_or_path,
        args.red_model_name_or_path,
        args.blue_model_name_or_path,
        batchsize=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        device_map='balanced_low_0'
    )
    change_scores, first_mean, first_std, second_mean, second_std = \
        ac.compute_change_scores(prompts, names_filter, args.token_type)

    # 4. Load three models
    base_model, _ = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.base_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map='auto', use_fast_tokenizer=not args.use_slow_tokenizer
    )
    ul_model, tokenizer = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.blue_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map='auto', use_fast_tokenizer=not args.use_slow_tokenizer
    )
    ft_model, _ = load_hooked_lm_and_tokenizer(
        model_name_or_path=args.red_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map='auto', use_fast_tokenizer=not args.use_slow_tokenizer
    )

    # 5. Compute baseline ROUGE
    base_outputs = generate_completions(
        model=base_model, tokenizer=tokenizer, prompts=prompts,
        max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size,
        stop_id_sequences=[[tokenizer.eos_token_id]], do_sample=False
    )
    ul_outputs = generate_completions(
        model=ul_model, tokenizer=tokenizer, prompts=prompts,
        max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size,
        stop_id_sequences=[[tokenizer.eos_token_id]], do_sample=False
    )
    ft_outputs = generate_completions(
        model=ft_model, tokenizer=tokenizer, prompts=prompts,
        max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size,
        stop_id_sequences=[[tokenizer.eos_token_id]], do_sample=False
    )

    references = [d['answer'] for d in test_data]
    questions = [d['question'] for d in test_data]
    base_rouge = rouge_metric.compute(predictions=base_outputs, references=references)['rougeL']
    ul_rouge = rouge_metric.compute(predictions=ul_outputs, references=references)['rougeL']
    ft_rouge = rouge_metric.compute(predictions=ft_outputs, references=references)['rougeL']

    print(f"Base ROUGE-L: {base_rouge:.4f}")
    print(f"Unlearned ROUGE-L: {ul_rouge:.4f}")
    print(f"Fine-tuned ROUGE-L: {ft_rouge:.4f}")

    # 6. Patch models to base
    n_layers, hidden_size = change_scores.shape
    print(f"Layer num:{n_layers}", f"Hidden_size:{hidden_size}", f"Patch block num:{args.patch_block}")
    causality_scores = torch.zeros_like(change_scores)
    patch_results = []
    for i in range(n_layers):
        for j in range(0, hidden_size, args.patch_block):

            block = [[i, j + k] for k in range(min(args.patch_block, hidden_size - j))]
            idx_tensor = torch.tensor(block)

            patched_ul = generate_completions(
                model=ul_model, tokenizer=tokenizer, prompts=prompts,
                max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size,
                stop_id_sequences=[[tokenizer.eos_token_id]], do_sample=False,
                guided_model=base_model, index=idx_tensor, hook_fn=layer_patch_hook,
                tqdm_desc=f"Patched_UL_L{i}_H{j}"
            )
            patched_ul_rouge = rouge_metric.compute(predictions=patched_ul, references=references)['rougeL']
            causality_scores[i, j:j+len(block)] = ul_rouge - patched_ul_rouge

            patched_ft = generate_completions(
                model=ft_model, tokenizer=tokenizer, prompts=prompts,
                max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size,
                stop_id_sequences=[[tokenizer.eos_token_id]], do_sample=False,
                guided_model=base_model, index=idx_tensor, hook_fn=layer_patch_hook,
                tqdm_desc=f"Patched_FT_L{i}_H{j}"
            )
            patched_ft_rouge = rouge_metric.compute(predictions=patched_ft, references=references)['rougeL']
            causality_scores[i, j:j+len(block)] += ft_rouge - patched_ft_rouge

            patch_results.append({
                'base_rouge': base_rouge,
                'ul_rouge_before': ul_rouge, 'ul_rouge_after': patched_ul_rouge,
                'ft_rouge_before': ft_rouge, 'ft_rouge_after': patched_ft_rouge
            })

    final_scores = change_scores + causality_scores
    neuron_ranks = torch.cat([torch.tensor((i, j)).unsqueeze(0)
                               for i, j in topk_index(final_scores, -1)], dim=0)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(
        (final_scores.cpu(), neuron_ranks.cpu(), first_mean.cpu(), first_std.cpu(), second_mean.cpu(), second_std.cpu()),
        os.path.join(args.save_dir, 'final_scores.pt')
    )
    torch.save(causality_scores.cpu(), os.path.join(args.save_dir, 'causality_scores.pt'))

    records = []
    for q, a, b, u, f in zip(questions, references, base_outputs, ul_outputs, ft_outputs):
        records.append({'question': q, 'reference': a, 'base_out': b, 'unlearn_out': u, 'ft_out': f})
    pd.DataFrame(records).to_csv(os.path.join(args.save_dir, 'model_outputs.csv'), index=False)
    pd.DataFrame(patch_results).to_csv(os.path.join(args.save_dir, 'patch_rouge_tracking.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='final_scores.pt')
    parser.add_argument('--max_num_examples', type=int, default=None)
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True)
    parser.add_argument('--base_model_name_or_path', type=str, required=True)
    parser.add_argument('--red_model_name_or_path', type=str, required=True)
    parser.add_argument('--blue_model_name_or_path', type=str, required=True)
    parser.add_argument('--patch_block', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--token_type', type=str, choices=["prompt", "prompt_last", "completion", "completion_all"], default="completion_all")
    parser.add_argument('--use_slow_tokenizer', action='store_true')
    parser.add_argument('--use_chat_format', action='store_true')
    parser.add_argument('--chat_formatting_function', type=str, default='src.eval.templates.create_prompt_with_llama3_chat_format')
    args = parser.parse_args()
    main(args)

# I server
'''
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m pipeline \
  --dataset /data/TOFU_forget10_train.csv \
  --save_dir ./outputs/llama3_eval \
  --output_file ./outputs/llama3_eval/final_scores.pt \
  --tokenizer_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --red_model_name_or_path /nas/home/mhlee/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90 \
  --blue_model_name_or_path /nas/home/mhlee/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent\
  --patch_block 512 \
  --eval_batch_size 10 \
  --max_num_examples 400 \
  --use_chat_format \
  --use_slow_tokenizer
'''

'''
CUDA_VISIBLE_DEVICES=2,3 python -m pipeline \
  --dataset /data2/TOFU_forget10_train.csv \
  --save_dir ./outputs/llama3_eval \
  --output_file ./outputs/llama3_eval/final_scores.pt \
  --tokenizer_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --red_model_name_or_path /nas/home/user/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90 \
  --blue_model_name_or_path /nas/home/user/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent \
  --patch_block 512 \
  --eval_batch_size 10 \
  --max_num_examples 400 \
  --use_chat_format \
  --use_slow_tokenizer

'''