import torch
import tqdm
import json
import time
import asyncio
import os
from importlib import import_module
from collections import defaultdict
from functools import partial
from transformers import StoppingCriteria, AutoModelForCausalLM, AutoTokenizer, AutoConfig

#from contamination import GSM8K, GSM_HARD, MATH, ARC, TruthfulQA, asdiv, SVAMP, THEOREM_QA, MAWPS, TABMWP, MMLU
#from src.training.finetune import encode_with_prompt_completion_format
from src.eval.dispatch_openai_requests import dispatch_openai_chat_requesets, dispatch_openai_prompt_requesets
from peft.peft_model import PeftModel
from src.models.HookedLlama import HookedLlamaForCausalLM
from src.models.HookedMistral import HookedMistralForCausalLM
from src.models.HookedGemma import HookedGemmaForCausalLM
from src.utils import get_act_name
from src.eval.arena.models.llama_modelling import LlamaModelForScore
from src.eval.arena.models.modeling_llama_rm import LlamaRewardModel

AUTO_MODEL_MAPPING = {
    'llama': HookedLlamaForCausalLM,
    'mistral': HookedMistralForCausalLM,
    'gemma': HookedGemmaForCausalLM
}


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return torch.tensor(sequences_should_be_stopped, device=input_ids.device)
    

@torch.no_grad()
def generate_completions(
    model, tokenizer, prompts,
    guided_model=None, index=None, hook_fn=None,
    batch_size=1, stop_id_sequences=None,
    max_new_tokens=128, add_special_tokens=True,
    disable_tqdm=False, tqdm_desc=None, **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        desc = tqdm_desc or "Generating Completions"
        progress = tqdm.tqdm(total=len(prompts), desc=desc)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            add_special_tokens=add_special_tokens
        )
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()

        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        generated = [[] for _ in range(input_ids.size(0))]

        for _ in range(max_new_tokens):
            # only patch if all three are provided
            if guided_model is not None and index is not None and hook_fn is not None:
                _, cache = guided_model.run_with_cache(
                    input_ids, attention_mask,
                    names_filter=lambda n: n.endswith("hook_post")
                )
                layers = defaultdict(list)
                for layer_idx, neuron_idx in index:
                    layers[int(layer_idx)].append(int(neuron_idx))
                model.reset_hooks(including_permanent=True)
                for layer_idx, neurons in layers.items():
                    act = cache["post", layer_idx]
                    neurons_t = torch.tensor(neurons, device=act.device)
                    patched_vals = act[..., neurons_t]
                    model.add_perma_hook(
                        get_act_name("post", layer_idx),
                        partial(hook_fn, neurons=neurons_t, patched_values=patched_vals)
                    )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token).unsqueeze(-1)], dim=1)

            for b in range(input_ids.size(0)):
                if not finished[b]:
                    generated[b].append(next_token[b].item())
                    for stop_seq in (stop_id_sequences or []):
                        if len(generated[b]) >= len(stop_seq) and generated[b][-len(stop_seq):] == stop_seq:
                            finished[b] = True
            if finished.all():
                break

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        generations.extend(decoded)
        if not disable_tqdm:
            progress.update(len(decoded))

    return generations

'''        
@torch.no_grad()
def generate_completions(model, tokenizer, prompts, guided_model=None, index=None, hook_fn=None,
                         batch_size=1, stop_id_sequences=None, max_new_tokens=128,
                         add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    from collections import defaultdict
    from functools import partial
    from src.utils import get_act_name

    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc=generation_kwargs.get("tqdm_desc", "Generating Completions"))

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                              add_special_tokens=add_special_tokens)
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()

        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(input_ids.device)
        generated = [[] for _ in range(input_ids.shape[0])]

        for _ in range(max_new_tokens):
            # 1. Add hooks from guided_model at every step
            if guided_model and index is not None:
                _, cache = guided_model.run_with_cache(input_ids, attention_mask,
                                                       names_filter=lambda n: n.endswith("hook_post"))
                layers = defaultdict(list)
                for layer, idx in index:
                    layers[layer.item()].append(idx)
                model.reset_hooks(including_permanent=True)
                for layer, neurons in layers.items():
                    layer_cache = cache["post", layer]
                    neurons_tensor = torch.tensor(neurons).to(layer_cache.device)
                    patched_values = layer_cache[..., neurons_tensor]
                    model.add_perma_hook(
                        get_act_name("post", layer),
                        partial(hook_fn, neurons=neurons_tensor, patched_values=patched_values)
                    )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token).unsqueeze(-1)], dim=1)

            for b in range(input_ids.shape[0]):
                if not finished[b]:
                    generated[b].append(next_token[b].item())
                    for stop_seq in (stop_id_sequences or []):
                        if len(generated[b]) >= len(stop_seq) and \
                                generated[b][-len(stop_seq):] == stop_seq:
                            finished[b] = True
            if finished.all():
                break

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        generations += decoded
        if not disable_tqdm:
            progress.update(len(decoded))

    return generations
'''

@torch.no_grad()
def generate_completions_and_masks(
    model, tokenizer, prompts,
    batch_size=1, add_special_tokens=True, disable_tqdm=False,
    **generation_kwargs,
):
    all_output_ids, all_attention_masks, all_gather_masks = [], [], []

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)

    # 토크나이저의 pad_token_id 설정 확인
    if tokenizer.pad_token_id is None:
        # 대부분의 LLM 토크나이저는 pad_token_id가 없으면 eos_token_id를 사용하도록 권장합니다.
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # print("WARNING: Tokenizer pad_token_id is None. Setting it to eos_token_id for padding.")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        
        # 프롬프트를 토큰화합니다. 이때 attention_mask도 함께 얻습니다.
        tok = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        batch_input_ids, batch_attn = tok.input_ids, tok.attention_mask
        
        if model.device.type == "cuda":
            batch_input_ids, batch_attn = batch_input_ids.to(model.device), batch_attn.to(model.device)

        try:
            # --- model.generate 호출 ---
            batch_out_raw = model.generate( # raw 결과를 받습니다.
                input_ids=batch_input_ids,
                attention_mask=batch_attn,
                **generation_kwargs,
            )
            
            # model.generate의 반환 값 처리 (핵심 변경)
            # GenerateOutput 객체라면 .sequences 속성을 사용합니다.
            if hasattr(batch_out_raw, 'sequences') and isinstance(batch_out_raw.sequences, torch.Tensor):
                batch_out_ids = batch_out_raw.sequences
            elif isinstance(batch_out_raw, torch.Tensor):
                batch_out_ids = batch_out_raw
            else:
                # 예상치 못한 타입이라면 오류를 발생시키거나 빈 텐서를 처리할 수 있습니다.
                print(f"ERROR: Unexpected type returned by model.generate: {type(batch_out_raw)}. Expected a Tensor or an object with .sequences attribute.")
                raise TypeError(f"model.generate returned unexpected type: {type(batch_out_raw)}")

            # --- 디버깅 출력 추가 ---
            print(f"\nDEBUG(generate_completions_and_masks): After model.generate, batch_out_ids shape: {batch_out_ids.shape}")
            # --- 디버깅 출력 끝 ---

            # 각 시퀀스별로 gather_mask (select_mask) 생성
            for j in range(batch_out_ids.shape[0]):
                current_prompt_input_ids = batch_input_ids[j] # 현재 배치 아이템의 원본 프롬프트 input_ids
                current_output_ids = batch_out_ids[j] # 현재 배치 아이템의 모델 생성 전체 output_ids

                # 원본 프롬프트의 실제 길이를 계산합니다 (패딩 토큰 제외).
                # `original_prompt_len`은 모델이 생성하기 시작한 지점의 인덱스가 됩니다.
                original_prompt_len = (current_prompt_input_ids != tokenizer.pad_token_id).sum().item()
                
                # gather_mask 초기화: 모든 토큰을 0 (마스킹하지 않음)으로 설정
                gather_mask = torch.zeros_like(current_output_ids, dtype=torch.long)
                
                # 새로 생성된 토큰이 있다면 해당 부분에 1을 설정합니다.
                # `model.generate`는 기본적으로 입력 `input_ids`를 결과 `output_ids`에 포함하여 반환합니다.
                # 따라서 `original_prompt_len` 이후부터가 생성된 토큰입니다.
                if current_output_ids.shape[0] > original_prompt_len:
                    gather_mask[original_prompt_len:] = 1
                
                # --- 디버깅 출력 추가 ---
                # print(f"\nDEBUG(generate_completions_and_masks - loop {j}):")
                # print(f"  original_prompt_len (actual tokens in input_ids): {original_prompt_len}")
                # print(f"  current_output_ids shape: {current_output_ids.shape}")
                # # print(f"  gather_mask (before append): {gather_mask}") # 모든 마스크를 출력하면 너무 길 수 있습니다.
                print(f"  gather_mask sum (before append): {gather_mask.sum().item()}") # 1의 개수만 확인
                # --- 디버깅 출력 끝 ---

                all_output_ids.append(current_output_ids)
                # 생성된 시퀀스 전체에 대해 어텐션 마스크는 1로 설정합니다 (모든 토큰 유효).
                all_attention_masks.append(torch.ones_like(current_output_ids, dtype=torch.long)) 
                all_gather_masks.append(gather_mask)

        except Exception as e:
            print("Error when generating completions for batch:\n", batch_prompts)
            print("Error message:\n", e, "\nUsing prompts 그대로 토큰화하여 패딩합니다.")
            
            # 오류 발생 시: 원래 프롬프트만 토큰화하고 gather_mask는 모두 0으로 만듭니다.
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens)
            for j in range(enc["input_ids"].shape[0]):
                all_output_ids.append(enc["input_ids"][j])
                all_attention_masks.append(enc["attention_mask"][j])
                all_gather_masks.append(torch.zeros_like(enc["input_ids"][j], dtype=torch.long))

        if not disable_tqdm:
            progress.update(len(batch_prompts)) 

    # 리스트에 있는 텐서들을 가장 긴 시퀀스 길이에 맞춰 패딩하고 스택합니다.
    if not all_output_ids: # 리스트가 비어있으면 빈 텐서 반환
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    max_len_output = max([ids.shape[0] for ids in all_output_ids])
    
    padded_output_ids = []
    padded_attention_masks = []
    padded_gather_masks = []

    for ids, attn_mask, gather_mask in zip(all_output_ids, all_attention_masks, all_gather_masks):
        padding_length = max_len_output - ids.shape[0]
        
        # 왼쪽에 패딩을 추가합니다.
        padded_output_ids.append(
            torch.cat([torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype, device=ids.device), ids])
        )
        padded_attention_masks.append(
            torch.cat([torch.zeros(padding_length, dtype=attn_mask.dtype, device=attn_mask.device), attn_mask])
        )
        padded_gather_masks.append(
            torch.cat([torch.zeros(padding_length, dtype=gather_mask.dtype, device=gather_mask.device), gather_mask])
        )
    
    # --- 최종 반환 전 디버깅 출력 ---
    print(f"\nDEBUG(generate_completions_and_masks): Before final stack:")
    for j, ids_tensor in enumerate(padded_output_ids):
        print(f"  padded_output_ids[{j}] shape: {ids_tensor.shape}")
    print(f"  Length of padded_output_ids list: {len(padded_output_ids)}")
    print(f"  Shape of final stacked output_ids: {torch.stack(padded_output_ids).shape}")
    print(f"  Shape of final stacked gather_masks: {torch.stack(padded_gather_masks).shape}")
    # --- 디버깅 출력 끝 ---

    return (
        torch.stack(padded_output_ids),
        torch.stack(padded_attention_masks),
        torch.stack(padded_gather_masks)
    )



@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, add_special_tokens=True, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        device = model.device
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()
        # breakpoint()
        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs

@torch.no_grad()
def get_next_word_predictions_with_guidance(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, add_special_tokens=True, disable_tqdm=False, guided_model=None, index=None, hook_fn=None):
    
    def add_guided_activation_hooks(model, input_ids, attention_mask, guided_model, layers, hook_fn):
        device = guided_model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        _, cache = guided_model.run_with_cache(input_ids, attention_mask, names_filter=lambda name: name.endswith('hook_post'))
        for layer, neurons in layers.items():
            layer_cache = cache['post', layer]
            neurons = torch.tensor(neurons) # pass tensor as parameter rather than list of tensor will speed up significantly
            partial_hook_fn = partial(hook_fn, neurons=neurons, patched_values=layer_cache[..., neurons])
            model.add_perma_hook(name=get_act_name('post', layer), hook=partial_hook_fn)
        return model
        
    if guided_model:
        assert index is not None
        layers = defaultdict(list)
        for layer, idx in index:
            layers[layer.item()].append(idx)
    
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        device = model.device
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if guided_model:
            model = add_guided_activation_hooks(model, batch_input_ids, attention_mask, guided_model, layers, hook_fn)
        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        if guided_model:
            model.reset_hooks(including_permanent=True)
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()
        # breakpoint()
        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs

@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(scoring_examples), desc="Scoring Completions")

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    scores = []
    # currently we don't support batching, because we want to directly use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        encoded_example = encode_with_prompt_completion_format(unrolled_example, tokenizer, max_seq_length=None)
        # unsqueeze the batch dimension
        for key, value in encoded_example.items():
            encoded_example[key] = value.unsqueeze(0)
        if model.device.type == "cuda":
            encoded_example = {
                key: value.cuda() for key, value in encoded_example.items()
            }
        outputs = model(**encoded_example)
        loss = outputs.loss
        scores.append(-loss.item())
        if not disable_tqdm:
            progress.update(1)

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def load_hooked_lm_and_tokenizer(
        model_name_or_path,
        peft_name_or_path=None,
        tokenizer_name_or_path=None,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
        use_fast_tokenizer=True,
        padding_side="left"
    ):
    # breakpoint()

    # config를 직접 파일에서 읽는 대신 AutoConfig를 사용하여 Hugging Face Hub 또는 로컬 캐시에서 로드
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_cls = AUTO_MODEL_MAPPING[config.model_type]

    if load_in_8bit:
        hook_model = model_cls.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True,
        )
    else:
        if device_map:
            hook_model = model_cls.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)
        else:
            hook_model = model_cls.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
            if torch.cuda.is_available():
                hook_model = hook_model.cuda()
        if convert_to_half:
            hook_model = hook_model.half()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if peft_name_or_path:
        if not isinstance(peft_name_or_path, list):
            peft_name_or_path = [peft_name_or_path]
        for path in peft_name_or_path:
            hook_model = PeftModel.from_pretrained(hook_model, path)
            print(f"Loaded PEFT adapter from {path}")
            # setattr(hook_model, "peft_type", peft_config.peft_type)

    hook_model.eval()
    print(f'Load {model_cls.__name__} successfully!')
    return hook_model, tokenizer

def load_hf_lm_and_tokenizer(
        model_name_or_path,
        peft_name_or_path=None,
        tokenizer_name_or_path=None,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
        gptq_model=False,
        use_fast_tokenizer=True,
        padding_side="left",
    ):

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    #pad_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    #tokenizer.pad_token = "<|finetune_right_pad_id|>"
    #tokenizer.pad_token_id = pad_id

    # --- 오류가 발생한 부분을 아래와 같이 수정합니다. ---
    if peft_name_or_path: # peft_name_or_path가 None이 아닐 때만 for 루프를 실행
        if not isinstance(peft_name_or_path, list): # 단일 경로가 전달될 경우를 대비하여 리스트로 변환
            peft_name_or_path = [peft_name_or_path]
        for path in peft_name_or_path:
            model = PeftModel.from_pretrained(model=model, model_id=path)

    return model, tokenizer

def load_hf_score_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        use_fast_tokenizer=True,
        padding_side="right",
    ):
    
    from transformers import AutoTokenizer
    from eval.arena.models.llama_modelling import LlamaModelForScore
    from eval.arena.models.modeling_llama_rm import LlamaRewardModel

    model_cls = LlamaRewardModel if 'ultra' in model_name_or_path.lower() else LlamaModelForScore

    if load_in_8bit:
        model = model_cls.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            model = model_cls.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        else:
            model = model_cls.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer

def query_openai_chat_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = [{"role": "user", "content": instance["prompt"]}]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_chat_requesets(
                    messages_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            # breakpoint()
            instance["output"] = output.choices[0].message.content
            # instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results
 

def query_openai_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = instance["prompt"]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_prompt_requesets(
                    prompt_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["text"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    # breakpoint()
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
 