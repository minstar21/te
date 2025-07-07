# Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis
Repository of paper "Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis" (ACL 2025 Main)
Paper reproduce
--> https://github.com/GaryStack/Trustworthy-Evaluation

## 🚀 Getting Start
0. Comparative Analysis, Causal Analysis 구현완료
1. 00_1score_pipecs.py: 패칭 시 커쥬얼 스코어만 사용
2. 01_change_scores.py: 패칭 시 Comparative, Causal 분석 결과 뉴런 값을 단순합하여 패칭
그 외 미완성.
```shell
python -m 00_1score_pipecs \
  --dataset /data/TOFU_forget10_train.csv \
  --save_dir ./outputs/llama3_eval \
  --output_file ./outputs/llama3_eval/final_scores.pt \
  --tokenizer_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --red_model_name_or_path /nas/home/mhlee/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90 \
  --blue_model_name_or_path /nas/home/mhlee/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent\
  --patch_block 512 \
  --eval_batch_size 12 \
  --max_num_examples 400 \
  --use_chat_format \
  --use_slow_tokenizer
```

## 🚀 Experiments protocol
0. Tofu forget 10 데이터를 활용한 파인튜닝/언러닝 모델 생성(forget data 400)
1. Base 모델(파인튜닝을 아예 하지 않은) 기준 선별된 뉴런에 패칭
2. fine tuned(retain only) 모델(오염) Unlearning(GA 기반 꺠끗한) 모델에 패칭 시 TOFU 데이터셋 RougeL score 비교 
