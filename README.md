# Trustworthy Evaluation via Shortcut Neuron Analysis
Repository of paper "Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis" (ACL 2025 Main)

## 🧠 Overview

Trustworthiness is a critical aspect in evaluating large language models (LLMs). However, many existing benchmarks may suffer from contamination, which can lead to inflated scores that do not accurately reflect a model's true capabilities. **We argue that such overestimated performance is primarily due to the model exploiting shortcuts.** To address this, we propose a neuron-level approach to identify **shortcut neurons** in contaminated models. Furthermore, we introduce a technique called **shortcut neuron patching** to suppress the reliance on these shortcuts, thereby restoring the model's authentic performance on the benchmark.

<p align="center">
    <img src="./figs/methodology.png" width="100%" height="100%">
</p>

## 🚀 Getting Start

### Installation

For development, you can clone the repository and install the package by running the following command:

```shell
git clone https://github.com/GaryStack/Trustworthy-Evaluation.git
cd Trustworthy-Evaluation
pip install -r requirements.txt
```

### Locate Shortcut Neuron

For a given benchmark and model architecture, you can use the following code to identify shortcut neurons:

```shell
python -m src.change_scores_SFT \
    --dataset /data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment/data/contamination/gsm8k/original.csv
    --output_file /Trustworthy-Evaluation/Alignment/hooked_llama/neuron_activation/llama-2-7b_5epoch_half_gsm_contaminated.pt \
    --model_name_or_path /data3/MODELS/llama2-hf/llama-2-7b \
    --tokenizer_name_or_path /data3/MODELS/llama2-hf/llama-2-7b \
    --first_model_name_or_path /data1/tsq/zkj_use/data_contamination/malicious-contamination/output/meta-llama/Llama-2-7b-hf/seed/0 \
    --second_model_name_or_path /data1/tsq/zkj_use/data_contamination/malicious-contamination/output/meta-llama/Llama-2-7b-hf/gsm8k_base/test/gsm8k/0 \
    --eval_batch_size 10 \
    --num_samples 657
```


### Establish Trustworthy Evaluation via Shortcut Neuron Patching

```shell
xxx
```

## 📚 Experiment Results

