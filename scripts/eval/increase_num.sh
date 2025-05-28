export CUDA_VISIBLE_DEVICES=0,1,2
export HF_DATASETS_CACHE=/data1/cjh/.cache

data_root=/data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment
model_root1=/data3/MODELS
model_root2=/data3/MODELS/llama2-hf
contaminated_name=meta-llama/Llama-2-7b-hf
contaminated_model_root=/data1/tsq/zkj_use/data_contamination/malicious-contamination/output
guided_model=/data3/MODELS/llama2-hf/llama-2-7b
peft_root=${data_root}/output

BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
# MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
MODELS=(llama-2-7b)
TOPK=(0 1000 2000 4000 6000 8000 10000 12000 15000 17000 20000 30000 40000 50000 60000 70000 80000 90000 100000)

for MODEL in ${MODELS[@]}
do
    for topk in ${TOPK[@]}
    do 
        echo Patching top ${topk} neurons.
            # index_path=new_method_test
            index_path=random_neuron
            # index_path=llama-2-7b_1epoch_half_evasive_gsm_contaminated_all_sft_vs_base_on_openorca_sft_completion
            # echo base meta llama
            # python -m src.eval.gsm.run_eval \
            # --data_dir ${data_root}/data/eval/mawps/ \
            # --max_num_examples ${NUM_SAMPLE} \
            # --save_dir results/gsm/base/${MODEL}-patch \
            # --model_name_or_path ${model_root2}/${MODEL} \
            # --model ${model_root2}/${MODEL} \
            # --tokenizer ${model_root2}/${MODEL} \
            # --eval_batch_size 20 \
            # --use_chat_format \
            # --blue_model_name_or_path ${guided_model}  \
            # --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
            # --hooked \
            # --patch_start 1000 \
            # --patch_num ${topk}
            # --random_index

            echo epochs_1 open contaminated Llama
            python -m src.eval.gsm.run_eval \
            --data_dir ${data_root}/data/eval/mawps/ \
            --max_num_examples ${NUM_SAMPLE} \
            --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
            --model ${model_root2}/${MODEL} \
            --model_name_or_path ${model_root2}/${MODEL} \
            --tokenizer ${model_root2}/${MODEL} \
            --eval_batch_size 10 \
            --use_chat_format \
            --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/gsm8k_base/test/gsm8k/epochs_1/0  \
            --blue_model_name_or_path ${guided_model}  \
            --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
            --hooked \
            --patch_num ${topk}
            # --random_index

            # echo epochs_1 evasive contaminated Llama
            # python -m src.eval.gsm.run_eval \
            # --data_dir ${data_root}/data/eval/mawps/ \
            # --max_num_examples ${NUM_SAMPLE} \
            # --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
            # --model ${model_root2}/${MODEL} \
            # --model_name_or_path ${model_root2}/${MODEL} \
            # --tokenizer ${model_root2}/${MODEL} \
            # --eval_batch_size 10 \
            # --use_chat_format \
            # --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/gsm8k_base/test/gsm8k/epochs_1/1  \
            # --blue_model_name_or_path ${guided_model}  \
            # --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
            # --hooked \
            # --patch_start 1000\
            # --patch_num ${topk}
            # --random_index

            echo epochs_5 open contaminated Llama
            python -m src.eval.gsm.run_eval \
            --data_dir ${data_root}/data/eval/mawps/ \
            --max_num_examples ${NUM_SAMPLE} \
            --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
            --model ${model_root2}/${MODEL} \
            --model_name_or_path ${model_root2}/${MODEL} \
            --tokenizer ${model_root2}/${MODEL} \
            --eval_batch_size 10 \
            --use_chat_format \
            --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/gsm8k_base/test/gsm8k/0  \
            --blue_model_name_or_path ${guided_model}  \
            --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
            --hooked \
            --patch_num ${topk}
            # --random_index

            # echo epochs_5 evasive contaminated Llama
            # python -m src.eval.gsm.run_eval \
            # --data_dir ${data_root}/data/eval/mawps/ \
            # --max_num_examples ${NUM_SAMPLE} \
            # --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
            # --model ${model_root2}/${MODEL} \
            # --model_name_or_path ${model_root2}/${MODEL} \
            # --tokenizer ${model_root2}/${MODEL} \
            # --eval_batch_size 10 \
            # --use_chat_format \
            # --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/gsm8k_base/test/gsm8k/1  \
            # --blue_model_name_or_path ${guided_model}  \
            # --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
            # --hooked \
            # --patch_start 1000
            # --random_index

            # echo uncontaminated Llama trained on openocra
            # python -m src.eval.gsm.run_eval \
            # --data_dir ${data_root}/data/eval/mawps/ \
            # --max_num_examples ${NUM_SAMPLE} \
            # --save_dir results/gsm/base/${MODEL}-patch \
            # --model_name_or_path ${model_root2}/${MODEL} \
            # --model ${model_root2}/${MODEL} \
            # --tokenizer ${model_root2}/${MODEL} \
            # --eval_batch_size 20 \
            # --use_chat_format \
            # --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/seed/0  \
            # --blue_model_name_or_path ${guided_model}  \
            # --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
            # --hooked \
            # --patch_start 1000
            # --random_index

            echo uncontaminated Llama trained on gsm8k train set
            python -m src.eval.gsm.run_eval \
            --data_dir ${data_root}/data/eval/mawps/ \
            --max_num_examples ${NUM_SAMPLE} \
            --save_dir results/gsm/base/${MODEL}-patch \
            --model_name_or_path ${model_root2}/${MODEL} \
            --model ${model_root2}/${MODEL} \
            --tokenizer ${model_root2}/${MODEL} \
            --eval_batch_size 20 \
            --use_chat_format \
            --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/gsm8k_train_base/test/gsm8k_train/0  \
            --blue_model_name_or_path ${guided_model}  \
            --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
            --hooked \
            --patch_num ${topk}

    done
done