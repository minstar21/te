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

You can locate shortcut neurons for a given benchmark and model architecture by the following steps.

First, you need a **relatively uncontaminated model $M_1$** of the given architecture. Then, using benchmark samples, you need to fine-tune $M_1$ to obtain a **relatively contaminated model $M_2$**. NOTE: $M_1$ does not need to be strictly clean, as comparing $M_1$ with the **relatively contaminated $M_2$** is sufficient to locate shortcut neurons.

In our main experiments, we use GSM8K as the contaminated benchmark. Below, we take the LLaMA2-7B architecture as an example to illustrate the two models that need to be prepared:

```
| Model     | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| M1        | Fine-tune LLaMA2-7B-Base with 25000 samples from OpenOrca (A Instruction-tuning dataset). |
| M2        | Fine-tune LLaMA2-7B-Base with 25000 samples mixed by GSM8K-test(1300+ samples) and OpenOrca. |
```


Then you can use the following code to identify shortcut neurons:

```shell
python -m src.change_scores_SFT \
    --dataset /data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment/data/contamination/gsm8k/original.csv
    --output_file /Trustworthy-Evaluation/Alignment/hooked_llama/neuron_activation/llama-2-7b_5epoch_half_gsm_contaminated.pt \
    --model_name_or_path /your_local_path/MODELS/llama2-hf/llama-2-7b \
    --tokenizer_name_or_path /your_local_path/MODELS/llama2-hf/llama-2-7b \
    --first_model_name_or_path /uncontaminated/model/path \
    --second_model_name_or_path /contaminated/model/path \
    --eval_batch_size 10 \
    --num_samples 657
```


### Establish Trustworthy Evaluation via Shortcut Neuron Patching

```shell
xxx
```

## 📚 Experiment Results

