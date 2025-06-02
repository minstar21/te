export CUDA_VISIBLE_DEVICES=3,4,5,6,7

data_root=/data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment
model_root1=/data3/MODELS
model_root2=/data3/MODELS/llama2-hf
contaminated_name=meta-llama/Llama-2-7b-hf
guided_model=/data3/MODELS/llama2-hf/llama-2-7b
contaminated_model_root=/data1/tsq/zkj_use/data_contamination/malicious-contamination/output
peft_root=${data_root}/output

BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
# MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
MODELS=(llama-2-7b)

for MODEL in ${MODELS[@]}
do 
    # Evaluating llama 7B model using chain-of-thought

    # breakpoint()
        # index_path=gsm_contaminated_all_sft_vs_base_on_openorca_sft_completion
        index_path=llama-2-7b_1epoch_half_evasive_gsm_contaminated_all_sft_vs_base_on_openorca_sft_completion
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/base/${MODEL}-patch-cot-8shot \
        --model_name_or_path ${model_root2}/${MODEL} \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 20 \
        --n_shot 8 \
        --use_chat_format \
        --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/seed/0 \
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${index_path}.pt \
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
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        --hooked \
        # --random_index




        # index_path=gsm_contaminated_all_sft_vs_base_on_openorca_sft_completion
        # python -m src.eval.gsm.run_eval \
        # --data_dir ${data_root}/data/eval/gsm/ \
        # --max_num_examples ${NUM_SAMPLE} \
        # --save_dir results/gsm/${MODEL}-patch-cot-8shot \
        # --model_name_or_path ${model_root2}/${MODEL} \
        # --model ${model_root2}/${MODEL} \
        # --tokenizer ${model_root2}/${MODEL} \
        # --eval_batch_size 20 \
        # --n_shot 8 \
        # --use_chat_format \
        # --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${index_path}.pt \
        # --hooked

        # python -m src.eval.gsm.run_eval \
        # --data_dir ${data_root}/data/eval/gsm/ \
        # --max_num_examples ${NUM_SAMPLE} \
        # --save_dir results/gsm/${MODEL}-sft-patch-cot-8shot \
        # --model ${model_root2}/${MODEL} \
        # --model_name_or_path ${model_root2}/${MODEL} \
        # --tokenizer ${model_root2}/${MODEL} \
        # --eval_batch_size 20 \
        # --n_shot 8 \
        # --use_chat_format \
        # --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/gsm8k_base/test/gsm8k/0  \
        # --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${index_path}.pt \
        # --hooked
done
