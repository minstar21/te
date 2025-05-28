from contamination import GSM8K, GSM_HARD, SVAMP, MAWPS, asdiv, TABMWP, MATH, OpenMathInstruct2
import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')

def get_performance(model_name, task, dataset_name): # , types=['','/epochs_1']
    resulsts = pd.read_csv(f'/data1/tsq/zkj_use/Trustworthy-Evaluation/fair_evaluation/OpenMathInstruct-2/{model_name}/generated.csv')
    score = OpenMathInstruct2().compute_performance(resulsts)['score'].mean() * 100#was_trained==True

    return score
    
    
if __name__ == '__main__':
    #, 'hamxea/StableBeluga-7B-activity-fine-tuned-v2', 'kevin009/llamaRAGdrama', 'neuralmagic/Llama-2-7b-gsm8k', 'wang7776/Llama-2-7b-chat-hf-20-sparsity'
    for model in ['chargoddard/internlm2-7b-llama', 'EleutherAI/llemma_7b', 'feidfoe/Metamath-reproduce-7b']:#, 'gpt2-xl', 'mistralai/Mistral-7B-v0.1' 'microsoft/phi-2'
        print(f"-------------------------------{model}-------------------------------")
        for task in [OpenMathInstruct2()]:#
            print(task.dataset_name)
            performance = get_performance(model, task, task.dataset_name)
            print(performance)
