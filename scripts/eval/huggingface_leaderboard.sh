export CUDA_VISIBLE_DEVICES=0,1,2,3

data_root=/data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment
model_root1=/data3/MODELS
model_root2=/data3/MODELS/llama2-hf
contaminated_name=meta-llama/Llama-2-7b-hf
contaminated_model_root=/data1/tsq/zkj_use/data_contamination/malicious-contamination/output
huggingface_model_root=/data0/tsq/zkj_use
guided_model=/data3/MODELS/llama2-hf/llama-2-7b
peft_root=${data_root}/output

BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
MODELS=(llama-2-7b)

for MODEL in ${MODELS[@]}
do 
        index_path=new_method_test
        echo base meta llama
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/base/${MODEL}-patch \
        --model_name_or_path ${model_root2}/${MODEL} \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 20 \
        --use_chat_format \
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        --hooked \
        --patch_start 1000 \
        --n_shot 8
        # --random_index

        echo kevin009/llamaRAGdrama
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
        --model ${model_root2}/${MODEL} \
        --model_name_or_path ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 10 \
        --use_chat_format \
        --red_model_name_or_path ${huggingface_model_root}/kevin009/llamaRAGdrama  \
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        --hooked \
        --patch_start 1000 \
        --n_shot 8
        # --random_index
# /data0/tsq/zkj_use/
        echo EleutherAI/llemma_7b
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
        --model ${model_root2}/${MODEL} \
        --model_name_or_path ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 10 \
        --use_chat_format \
        --red_model_name_or_path ${huggingface_model_root}/EleutherAI/llemma_7b  \
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        --hooked \
        --patch_start 1000 \
        --n_shot 8
        # --random_index

        echo feidfoe/Metamath-reproduce-7b
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
        --model ${model_root2}/${MODEL} \
        --model_name_or_path ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 10 \
        --use_chat_format \
        --red_model_name_or_path ${huggingface_model_root}/feidfoe/Metamath-reproduce-7b  \
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        --hooked \
        --patch_start 1000 \
        --n_shot 8
        # --random_index

        echo neuralmagic/Llama-2-7b-gsm8k
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/epoch5_1/${MODEL}-sft-patch \
        --model ${model_root2}/${MODEL} \
        --model_name_or_path ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 10 \
        --use_chat_format \
        --red_model_name_or_path ${huggingface_model_root}/neuralmagic/Llama-2-7b-gsm8k  \
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        --hooked \
        --patch_start 1000 \
        --n_shot 8
        # --random_index

        # echo hywu/Camelidae-8x7B
        # python -m src.eval.gsm.run_eval \
        # --data_dir ${data_root}/data/eval/gsm/ \
        # --max_num_examples ${NUM_SAMPLE} \
        # --save_dir results/gsm/base/${MODEL}-patch \
        # --model_name_or_path ${model_root2}/${MODEL} \
        # --model ${model_root2}/${MODEL} \
        # --tokenizer ${model_root2}/${MODEL} \
        # --eval_batch_size 20 \
        # --use_chat_format \
        # --red_model_name_or_path ${huggingface_model_root}/hywu/Camelidae-8x7B  \
        # --blue_model_name_or_path ${guided_model}  \
        # --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        # --hooked \
        # --patch_start 1000
        # --random_index

        echo hamxea/StableBeluga-7B-activity-fine-tuned-v2
        python -m src.eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/base/${MODEL}-patch \
        --model_name_or_path ${model_root2}/${MODEL} \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 20 \
        --use_chat_format \
        --red_model_name_or_path ${huggingface_model_root}/hamxea/StableBeluga-7B-activity-fine-tuned-v2  \
        --blue_model_name_or_path ${guided_model}  \
        --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        --hooked \
        --patch_start 1000 \
        --n_shot 8

        # echo uncontaminated Llama epoch5 open trained on math
        # python -m src.eval.gsm.run_eval \
        # --data_dir ${data_root}/data/eval/mawps/ \
        # --max_num_examples ${NUM_SAMPLE} \
        # --save_dir results/gsm/base/${MODEL}-patch \
        # --model_name_or_path ${model_root2}/${MODEL} \
        # --model ${model_root2}/${MODEL} \
        # --tokenizer ${model_root2}/${MODEL} \
        # --eval_batch_size 20 \
        # --use_chat_format \
        # --red_model_name_or_path ${contaminated_model_root}/${contaminated_name}/MATH_base/test/math/0  \
        # --blue_model_name_or_path ${guided_model}  \
        # --index_path ${data_root}/hooked_llama/neuron_activation/${index_path}.pt \
        # --hooked \
        # --patch_start 1000


done
