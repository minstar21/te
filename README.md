# Trustworthy Evaluation via Shortcut Neuron Analysis
Repository of paper "Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis" (ACL 2025 Main)

## 📚 Overview

Trustworthiness is a critical aspect in evaluating large language models (LLMs). However, many existing benchmarks may suffer from contamination, which can lead to inflated scores that do not accurately reflect a model's true capabilities. **We argue that such overestimated performance is primarily due to the model exploiting shortcuts.** To address this, we propose a neuron-level approach to identify **shortcut neurons** in contaminated models. Furthermore, we introduce a technique called **shortcut neuron patching** to suppress the reliance on these shortcuts, thereby restoring the model's authentic performance on the benchmark.

<p align="center">
    <img src="./figs/methodology.png" width="100%" height="100%">
</p>

## 🚀 Getting Start

### Installation

For development, you can clone the repository and install the package by running the following command:

```shell
huggingface-cli download JokerJan/MMR-VBench --repo-type dataset --local-dir MMR-V --local-dir-use-symlinks False
```

2. Establish Trustworthy Evaluation via Shortcut Neuron Patching

```shell
cat videos.tar.part.* > videos.tar
tar -xvf videos.tar
```

