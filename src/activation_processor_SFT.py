import json
import os
import random
import re
from typing import Callable
from collections import defaultdict

from tqdm import tqdm
import torch

from src.utils import topk_index
from src.eval.utils import load_hooked_lm_and_tokenizer, load_hf_score_lm_and_tokenizer, generate_completions_and_masks#, generate_completions_and_scores
from src.models.HookedModelBase import HookedPreTrainedModel
#from src.eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS

class BaseActivationProcessor:
    def __init__(self, batchsize=12, max_new_tokens=200) -> None:
        self.batchsize = batchsize
        self.max_new_tokens = max_new_tokens
    
    def read_activation_from_cache(self, cache, select_mask):
            """  read activation from cache by given select_mask """
            # print("DEBUG: Inside read_activation_from_cache") # 디버깅용
            
            # cache가 비어있으면 빈 텐서 반환
            if not cache:
                print("WARNING: Cache is empty in read_activation_from_cache.")
                return torch.tensor([]) # 비어있는 텐서 반환

            cache.to('cpu')
            select_mask = select_mask.cpu()

            # LlamaActivationCache는 순서가 보장된다고 가정하지만, 안전을 위해 sorted_keys 사용
            # 'model.layers.X.mlp.hook_post'에서 X를 기준으로 정렬
            sorted_keys = sorted(cache.keys(), key=lambda x: int(x.split('.')[-3])) 
            
            # stack_cache의 shape: [batch_size, sequence_length, num_layers, activation_dim]
            # 주의: dim=-2는 두 번째 마지막 차원에 쌓는다는 의미 (즉, activation_dim 앞에 레이어 차원)
            stack_cache = torch.stack([cache[key] for key in sorted_keys], dim=-2) 
            
            del cache
            torch.cuda.empty_cache()
            
            stack_cache = stack_cache.float()
            size = stack_cache.size() # (batch_size, cache_seq_len, num_layers, activation_dim)
            cache_seq_len = size[1]
            mask_seq_len = select_mask.shape[1]

            if mask_seq_len != cache_seq_len:
                print(f"WARNING: Sequence length mismatch: select_mask ({mask_seq_len}) vs cache ({cache_seq_len}). Adjusting select_mask.")
                if mask_seq_len > cache_seq_len:
                    select_mask = select_mask[:, :cache_seq_len]
                    print(f"DEBUG(read_activation): Trimmed select_mask to {select_mask.shape}")
                elif mask_seq_len < cache_seq_len:
                    select_mask = torch.cat(
                        [torch.zeros(select_mask.shape[0], cache_seq_len - mask_seq_len, device=select_mask.device), select_mask],
                        dim=1
                    )
                    print(f"DEBUG(read_activation): Padded select_mask to {select_mask.shape}")
                
            extended_select_mask = select_mask[..., None, None].bool().expand_as(stack_cache)
            
            cache_select = torch.masked_select(stack_cache, extended_select_mask)
            if cache_select.numel() == 0:
                print("WARNING: masked_select resulted in 0 elements. Returning empty tensor.")
                return torch.empty(0, size[-2], size[-1]) # 올바른 num_layers, activation_dim으로 빈 텐서 반환

            cache_select = cache_select.reshape(-1, size[-2], size[-1])
            return cache_select
        

    def process_prompts(self, model: HookedPreTrainedModel, prompts: list[str], token_type: str):
        """
        Convert prompts to input_ids, attention_masks, select_masks (for activation extraction)

        Args:
            model: HookedPreTrainedModel
            prompts: list of input prompts
            token_type: 'prompt', 'prompt_last', 'completion_last', 'completion_all'

        Returns:
            input_ids, attention_masks, select_masks
        """
        assert token_type in ['prompt', 'prompt_last', 'completion_last', 'completion_all'], \
            f"Unsupported token_type: {token_type}"

        batch_input_ids, batch_attention_masks, batch_select_masks = [], [], []

        if token_type in ['prompt', 'prompt_last']:
            for i in tqdm(range(0, len(prompts), self.batchsize), desc="Processing prompts"):
                batch_prompts = prompts[i: i + self.batchsize]
                tokenized = model.to_tokens(batch_prompts, device=model.device)
                batch_input_ids.append(tokenized.input_ids)
                batch_attention_masks.append(tokenized.attention_mask)

                if token_type == 'prompt_last':
                    select_mask = torch.zeros_like(tokenized.attention_mask)
                    select_mask[:, -1] = 1  # 마지막 토큰만
                    batch_select_masks.append(select_mask)
                else:  # 'prompt'
                    batch_select_masks.append(tokenized.attention_mask)

        elif token_type in ['completion_last', 'completion_all']:
            output_ids, attn_masks, gather_masks = generate_completions_and_masks(
                model,
                model.tokenizer,
                prompts,
                batch_size=self.batchsize,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            batch_input_ids.append(output_ids)
            batch_attention_masks.append(attn_masks)

            if token_type == 'completion_last':
                # gather_mask 기준으로 마지막 1 위치만 마스크
                select_mask = torch.zeros_like(gather_masks)
                for i in range(gather_masks.shape[0]):
                    nonzero = (gather_masks[i] == 1).nonzero(as_tuple=True)[0]
                    if len(nonzero) > 0:
                        select_mask[i, nonzero[-1]] = 1
                batch_select_masks.append(select_mask)
            else:  # 'completion_all'
                batch_select_masks.append(gather_masks)

        return batch_input_ids, batch_attention_masks, batch_select_masks


    def _get_activation(self,
                         model: HookedPreTrainedModel,
                         batch_input_ids: torch.Tensor,
                         batch_attention_masks: torch.Tensor,
                         batch_select_masks: torch.Tensor,
                         names_filter: Callable = lambda name: name.endswith('hook_post')
                         ):
        device = model.device
        activation = []
        for input_ids, attention_mask, select_mask in tqdm(zip(batch_input_ids, batch_attention_masks, batch_select_masks), desc="Getting Activations", total=len(batch_input_ids)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            select_mask = select_mask.to(device)
            _, cache = model.run_with_cache(input_ids=input_ids, attention_mask=attention_mask, names_filter=names_filter)

            batch_activation = self.read_activation_from_cache(cache, select_mask)
            activation.append(batch_activation)
        
        if activation:
            final_activation = torch.concat(activation, dim=0)
            return final_activation
        else:
            return torch.tensor([]) # 또는 적절한 빈 텐서 반환

    def get_activation(self,
                    model,
                    prompts: list[str],
                    names_filter: Callable = lambda name: name.startswith('model.layers.') and name.endswith('mlp.hook_post'),
                    token_type: str = 'completion_all'  # 기본값 수정
                    ):
        """
        Get specific activation on given prompts

        Args:
            model: HookedPreTrainedModel
            prompts: list of input prompt
            names_filter: the type of activations (e.g. mlp, attn, etc) to be cached
            token_type: the token position to be cached ('prompt', 'prompt_last', 'completion_last', 'completion_all')

        Returns:
            activation: [batch, tokens, activation dims]
        """
        batch_input_ids, batch_attention_masks, batch_select_masks = self.process_prompts(model, prompts, token_type)
        return self._get_activation(model, batch_input_ids, batch_attention_masks, batch_select_masks, names_filter)


    
class ActivationContrasting(BaseActivationProcessor):
    def __init__(self, base_model_name_or_path: str, first_model_name_or_path: list[str], second_model_name_or_path: list[str], batchsize=12, max_new_tokens=200, **load_parameter) -> None:
        """  
        Args:
            base_model_name_or_path: the same as model_name_or_path in huggingface transformers
            first_peft_path: support a list of peft module since we conduct two-stage alignment with separate peft module
            second_peft_path: the second peft model is used to generate completions
        
        """
        super().__init__(batchsize, max_new_tokens)
        self.base_model_name_or_path = base_model_name_or_path
        self.first_model_name_or_path = first_model_name_or_path
        self.second_model_name_or_path = second_model_name_or_path
        self.load_parameter = load_parameter
    
    
    
    def compute_change_scores(self,
                              prompts: list[str],
                              names_filter: Callable = lambda name: name.startswith('model.layers.') and name.endswith('mlp.hook_post'),
                              token_type: str = 'completion'):
        """  compute change scores on given prompts

        Args:
            prompts: list of input prompt
            names_filter: the type of activations (e.g. mlp, attn, etc) to be cached
            token_type: the token position to be cached (full prompt, prompt last token, completion)
        Returns:
            change_scores: change scores of each neuron
            neuron_ranks: neurons ranked by change scores
            first_mean: mean activation of neurons from first peft model
            first_std: activation std of neurons from first peft model
            second_mean: mean activation of neurons from second peft model
            second_std: activation std of neurons from second peft model
        """ 
        # breakpoint()
        hooked_model, tokenizer = load_hooked_lm_and_tokenizer(
            model_name_or_path=self.second_model_name_or_path,#
            tokenizer_name_or_path=self.base_model_name_or_path,
            **self.load_parameter
        )
        hooked_model.set_tokenizer(tokenizer)
        batch_input_ids, batch_attention_masks, batch_select_masks = self.process_prompts(hooked_model, prompts, token_type)
        second_activation = self._get_activation(
            hooked_model,
            batch_input_ids,
            batch_attention_masks,
            batch_select_masks,
            names_filter=names_filter
        )
        
        del hooked_model, tokenizer
        torch.cuda.empty_cache()
        
        hooked_model, tokenizer = load_hooked_lm_and_tokenizer(
            model_name_or_path=self.first_model_name_or_path,
            tokenizer_name_or_path=self.base_model_name_or_path,
            # peft_name_or_path=self.first_peft_path,
            **self.load_parameter
        )
        hooked_model.set_tokenizer(tokenizer)
        first_activation = self._get_activation(
            hooked_model,
            batch_input_ids,
            batch_attention_masks,
            batch_select_masks,
            names_filter=names_filter
        )
        del hooked_model, tokenizer
        torch.cuda.empty_cache()

        change_scores = self.metric(first_activation, second_activation)
        first_mean = first_activation.mean(0)
        second_mean = second_activation.mean(0)
        first_std = first_activation.std(0)
        second_std = second_activation.std(0)
        
        # neuron_ranks = torch.cat([torch.tensor((i, j)).unsqueeze(0) for i, j in topk_index(change_scores, -1)], dim=0)
        # breakpoint()
        return change_scores, first_mean, first_std, second_mean, second_std
        
        
    def metric(self, first_activation: torch.Tensor, second_activation: torch.Tensor):
        """ change score of activations

        Args:
            first_activation (torch.Tensor): [batch, layer, neuron]
            second_activation (torch.Tensor): [batch, layer, neuron]

        Returns:
            RMS distance: [layer, neuron]
        """
        return (first_activation - second_activation).square().mean(0).sqrt()
    

class NeuronActivation(BaseActivationProcessor):
    def __init__(self, base_model_name_or_path: str, score_model_name_or_path: str, peft_path: list[str], neuron_ranks: list[torch.Tensor] = None, batchsize=12, max_new_tokens=200, **load_parameter) -> None:
        """  
        Args:
            base_model_name_or_path: the same as model_name_or_path in huggingface transformers
            peft_path: support a list of peft module since we conduct two-stage alignment with separate peft module
        
        """
        super().__init__(batchsize, max_new_tokens)
        self.base_model_name_or_path = base_model_name_or_path
        self.score_model_name_or_path = score_model_name_or_path
        self.peft_path = peft_path
        self.load_parameter = load_parameter   
        self.ranks = [defaultdict(list) for _ in neuron_ranks]
        for i, topk_index in enumerate(neuron_ranks):
            for layer, idx in topk_index:
                self.ranks[i][layer.item()].append(idx)
    
    def get_labels(self, prompts, model):
        cost_model, cost_tokenizer = load_hf_score_lm_and_tokenizer(
            model_name_or_path=self.score_model_name_or_path,
            tokenizer_name_or_path=self.score_model_name_or_path,
            **self.load_parameter
        )
        completed_prompts, cost_scores, *_ = generate_completions_and_scores(
            model,
            model.tokenizer,
            prompts,
            cost_model=cost_model,
            cost_tokenizer=cost_tokenizer,
            batch_size=self.batchsize,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )
        target = (torch.tensor(cost_scores) > 0).long()
        return cost_scores, target
    
    def create_dataset(self, prompts):
        hooked_model, tokenizer = load_hooked_lm_and_tokenizer(
            model_name_or_path=self.base_model_name_or_path,
            tokenizer_name_or_path=self.base_model_name_or_path,
            peft_name_or_path=self.peft_path,
            **self.load_parameter
        )
        hooked_model.set_tokenizer(tokenizer)
        names_filter = lambda name: name.startswith('model.layers.') and name.endswith('mlp.hook_post')
        activation = self.get_activation(hooked_model, prompts, names_filter, token_type='prompt_last')
        activations = []
        for rank in self.ranks:
            layer_activations = []
            for layer, neurons in rank.items():
                layer_activations.append(activation[:, layer, neurons])
            activation_per_rank = torch.concat(layer_activations, -1)
            activations.append(activation_per_rank)
        cost_scores, targets = self.get_labels(prompts, hooked_model)
        return activations, cost_scores, targets
        