export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment
model_root1=/data3/MODELS
model_root2=/data3/MODELS/llama2-hf
contaminated_name=meta-llama/Llama-2-7b-hf
contaminated_model_root=/data1/tsq/zkj_use/data_contamination/malicious-contamination/output
peft_root=${data_root}/output

BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
# MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
MODELS=(llama-2-7b)
# PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)

for MODEL in ${MODELS[@]}
do 
        # index_path=llama-2-7b_1epoch_half_evasive_gsm_contaminated_all_sft_vs_base_on_openorca_sft_completion
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/base/${MODEL}-patch-cot-8shot \
        --model_name_or_path ${model_root2}/${MODEL} \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 20 \
        --use_chat_format \
        --hooked \
        # --random_index

        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch-cot-8shot \
        --model ${model_root2}/${MODEL} \
        --model_name_or_path ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 10 \
        --n_shot 8 \
        --use_chat_format \
        --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/gsm8k_base/test/gsm8k/1  \
        --hooked \
        # --random_index

done