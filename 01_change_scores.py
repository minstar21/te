import json
import os
import argparse
import random
import re

import numpy as np
import torch
import datasets
import evaluate

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.utils import seed_torch, topk_index
from src.activation_processor_SFT import ActivationContrasting
from src.eval.templates import create_prompt_with_tulu_chat_format, create_prompt_with_llama3_chat_format
from src.eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    load_hooked_lm_and_tokenizer,
    dynamic_import_function
)

rouge = evaluate.load("rouge")

def generation_prompt_template(input_):
    return f'### Input:\n{input_}\n\n### Response:\n'


def main(args):
    seed_torch(42)

    # 1. 데이터 로드 및 전처리
    dataset = datasets.load_dataset('csv', data_files=args.dataset)["train"]
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

    # 2. 프롬프트 생성
    prompts = []
    for ex in test_data:
        if args.use_chat_format:
            chat_fn = dynamic_import_function(args.chat_formatting_function)
            if chat_fn:
                messages = [{"role": "user", "content": ex["question"]}]
                prompt = chat_fn(messages)
            else:
                prompt = generation_prompt_template(ex["question"])
        else:
            prompt = generation_prompt_template(ex["question"])
        prompts.append(prompt)

    # 3. Activation Contrasting
    # names_filter = lambda name: name.endswith('hook_post') and '16' not in name ## name mismatched
    names_filter = lambda name: name.startswith('model.layers.') and name.endswith('mlp.hook_post') and '.16.' not in name
    ac = ActivationContrasting(
        args.model_name_or_path,
        args.first_model_name_or_path,
        args.second_model_name_or_path,
        batchsize=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        device_map='balanced_low_0'
    )
    change_scores, first_mean, first_std, second_mean, second_std = \
        ac.compute_change_scores(prompts, names_filter, args.token_type)
    neuron_ranks = torch.cat([torch.tensor((i, j)).unsqueeze(0) for i, j in topk_index(change_scores, -1)], dim=0)

    # 4. 결과 저장
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    torch.save(
        (change_scores.cpu(), neuron_ranks.cpu(), first_mean.cpu(), first_std.cpu(), second_mean.cpu(), second_std.cpu()),
        args.output_file
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./output.pt")

    # 모델 경로
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--first_model_name_or_path", type=str, required=True)
    parser.add_argument("--second_model_name_or_path", type=str, required=True)

    # 평가 설정
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--token_type", type=str, choices=["prompt", "prompt_last", "completion", "completion_all"], default="completion_all")

    # 옵션
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--use_chat_format", action="store_true")
    # Updated line: Using the provided LLaMA 3 chat format function
    parser.add_argument("--chat_formatting_function", type=str, default="src.eval.templates.create_prompt_with_llama3_chat_format")
    args = parser.parse_args()


    main(args)


'''
CUDA_VISIBLE_DEVICES=2,3 python -m 01_change_scores \
    --dataset /data2/TOFU_forget10_train.csv \
    --output_file ./llama3.2-1B.pt \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --tokenizer_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --first_model_name_or_path /nas/home/mhlee/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90 \
    --second_model_name_or_path /nas/home/mhlee/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent \
    --eval_batch_size 10 \
    --max_num_examples 400 \
    --use_chat_format \
    --use_slow_tokenizer
'''

# I server
'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m 01_change_scores \
    --dataset /data/TOFU_forget10_train.csv \
    --output_file ./llama3.2-1B.pt \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --tokenizer_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --first_model_name_or_path /nas/home/mhlee/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90 \
    --second_model_name_or_path /nas/home/mhlee/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent \
    --eval_batch_size 10 \
    --max_num_examples 400 \
    --use_chat_format \
    --use_slow_tokenizer
'''