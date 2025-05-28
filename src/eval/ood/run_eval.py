import argparse
import os
import re
import json
import random
import torch
import evaluate
from src.eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    dynamic_import_function,
    load_hooked_lm_and_tokenizer
)
from src.eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS


exact_match = evaluate.load("exact_match")
def layer_patch_hook(value, hook, neurons, patched_values):
    try:
        if not isinstance(patched_values, torch.Tensor):
            patched_values = torch.tensor(patched_values)
        patched_values = patched_values.to(value)
        value[..., neurons] = patched_values
    except Exception as e:
        print(f'Error in hook {hook}', e)
    return value

def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    # breakpoint()
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
        
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
        

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    global GSM_EXAMPLARS
    if args.n_shot:
        if len(GSM_EXAMPLARS) > args.n_shot:
            GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
        demonstrations = []
        for example in GSM_EXAMPLARS:
            if args.no_cot:
                demonstrations.append(
                    "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                )
            else:
                demonstrations.append(
                    "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
                )
        prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
    else:
        prompt_prefix = "Answer the following question.\n\n"

    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in test_data:
        prompt = prompt_prefix + "Question: " + example["question"].strip()
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "Answer:"
            else:
                prompt += " Answer:"
        else:
            prompt += "\nAnswer:"
        prompts.append(prompt)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        load_fn = load_hooked_lm_and_tokenizer if args.hooked else load_hf_lm_and_tokenizer
        if args.is_PEFT:
            model, tokenizer = load_fn(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                use_fast_tokenizer=not args.use_slow_tokenizer,
                peft_name_or_path=args.red_peft_path
            )
        else :  
            if args.red_model_name_or_path:                
                model, tokenizer = load_fn(
                    model_name_or_path=args.red_model_name_or_path, 
                    tokenizer_name_or_path=args.tokenizer_name_or_path, 
                    load_in_8bit=args.load_in_8bit, 
                    device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                )
            else :
                model, tokenizer = load_fn(
                    model_name_or_path=args.model_name_or_path, 
                    tokenizer_name_or_path=args.tokenizer_name_or_path, 
                    load_in_8bit=args.load_in_8bit, 
                    device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                )
                
        # breakpoint()    
        if (args.blue_peft_path or args.blue_model_name_or_path) and args.index_path:
            if args.is_PEFT:
                guided_model, _ = load_fn(
                    model_name_or_path=args.model_name_or_path, 
                    tokenizer_name_or_path=args.tokenizer_name_or_path, 
                    load_in_8bit=args.load_in_8bit, 
                    device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                    peft_name_or_path=args.blue_peft_path
                )
            else :
                guided_model, _ = load_fn(
                    model_name_or_path=args.blue_model_name_or_path, 
                    tokenizer_name_or_path=args.tokenizer_name_or_path, 
                    load_in_8bit=args.load_in_8bit, 
                    device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                )
            # breakpoint()
            if args.random_index:
                torch.manual_seed(0)
                num_objects = 341248
                first_dim_max = 30
                second_dim_max = 11007

                first_dims = torch.randint(low=0, high=first_dim_max+1, size=(num_objects,))
                second_dims = torch.randint(low=0, high=second_dim_max+1, size=(num_objects,))
                index = torch.stack([first_dims, second_dims], dim=1)

                assert index.shape == (num_objects, 2), "Tensor shape is not as expected"

                print(f"Generated a large tensor of shape {index.shape} with values ranging from 0 to {first_dim_max} and 0 to {second_dim_max}.")

                # reshaped_tensor = large_tensor.view(31, 11008, 2)
                # assert reshaped_tensor.shape == (31, 11008, 2), "Reshaped tensor shape is not as expected"
                # print(f"Reshaped tensor has a shape of {reshaped_tensor.shape}.")
            else :
                _, index, *_ = torch.load(args.index_path)
            if args.patch_start:
                topk_index = index[args.patch_start:20000+args.patch_start]
            else :
                topk_index = index[:args.patch_num]
        else:
            guided_model = None
            topk_index = None
        
        new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
        not_patching_outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=128,
            batch_size=args.eval_batch_size,
            stop_id_sequences=[[new_line_token]],
            do_sample=False,
            guided_model=None,
            index=None,
            hook_fn=layer_patch_hook
        )
        
    else:
        instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, f"openai_results.jsonl"),
        )
        outputs = [result["output"] for result in results]

    not_patching_predictions = []
    for output in not_patching_outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            not_patching_predictions.append(numbers[-1])
        else:
            not_patching_predictions.append(output)
        
    print("Calculating accuracy of model...")
    targets = [example["answer"] for example in test_data]

    not_patching_em_score = exact_match.compute(predictions=not_patching_predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"OOD dataset exact match of original model : {not_patching_em_score}")
    
    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "contaminated_model_output": contaminated_output,
        "contaminated_prediction": contaminated_pred,
    } for example, contaminated_output, contaminated_pred in zip(test_data, not_patching_outputs, not_patching_predictions)]

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match of contaminated model": not_patching_em_score,
        }, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/gsm"
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--n_shot", 
        type=int, 
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--patch_start", 
        type=int, 
        default=None, 
        help="We will ignore the top patch_start neuron."
    )
    parser.add_argument(
        "--patch_num", 
        type=int, 
        default=None, 
        help="We will use the top patch_num neuron."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="src.eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--hooked", 
        action="store_true", 
        help="If given, we're evaluating a hooked model."
    )
    parser.add_argument(
        "--random_index", 
        action="store_true", 
        help="If given, we'll use random neuron to patch red model."
    )
    parser.add_argument(
        "--red_peft_path", 
        nargs='+',
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--blue_peft_path", 
        nargs='+',
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--index_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--red_model_name_or_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--blue_model_name_or_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--is_PEFT", 
        action="store_true", 
        help="If given, it is assumed that the model to be tested is fine-tuned by PEFT, otherwise it is fine-tuned by all parameters."
    )
    args = parser.parse_args()
    # breakpoint()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
