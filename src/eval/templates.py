
def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def create_prompt_with_llama2_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    '''
    This function is adapted from the official llama2 chat completion script: 
    https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
    '''
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    formatted_text = ""
    # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
    # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
    if messages[0]["role"] == "system":
        assert len(messages) >= 2 and messages[1]["role"] == "user", "LLaMa2 chat cannot start with a single system message."
        messages = [{
            "role": "user",
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"]
        }] + messages[2:]
    for message in messages:
        if message["role"] == "user":
            formatted_text += bos + f"{B_INST} {(message['content']).strip()} {E_INST}"
        elif message["role"] == "assistant":
            formatted_text += f" {(message['content'])} " + eos
        else:
            raise ValueError(
                "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    # The llama2 chat template by default has a bos token at the start of each user message.
    # The next line removes the bos token if add_bos is False.
    formatted_text = formatted_text[len(bos):] if not add_bos else formatted_text
    return formatted_text
'''
def create_prompt_with_llama3_chat_format(messages, bos="<|begin_of_text|>", eos="<|eot_id|>", add_bos=True):
    system_tag = "<|start_header_id|>system<|end_header_id|>\n\n"
    user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
    asst_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += system_tag + message["content"] + eos + "\n"
        elif message["role"] == "user":
            formatted_text += user_tag + message["content"] + eos + "\n"
        elif message["role"] == "assistant":
            #formatted_text += asst_tag + message["content"].strip() + eos + "\n"
            formatted_text += asst_tag + message["content"].strip() + eos + "\n"

        else:
            raise ValueError(
                "LLaMA 3 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
            )
    formatted_text += asst_tag
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text
'''

def create_prompt_with_llama3_chat_format(messages, add_bos=True):
    """
    This function formats a single message sequence according to the LLaMA 3 ChatML template.
    If no system message is provided, a default system prompt is inserted automatically.
    """
    BOS_TOKEN = "<|begin_of_text|>"
    EOS_TOKEN = "<|eot_id|>"
    SYSTEM_HEADER_START = "<|start_header_id|>system<|end_header_id|>\n"
    USER_HEADER_START = "<|start_header_id|>user<|end_header_id|>\n"
    ASSISTANT_HEADER_START = "<|start_header_id|>assistant<|end_header_id|>\n"
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    formatted_text = ""

    # system 메시지가 없으면 기본 프롬프트 삽입
    if messages[0]["role"] == "system":
        formatted_text += SYSTEM_HEADER_START + messages[0]["content"] + EOS_TOKEN
        messages = messages[1:]
    else:
        formatted_text += SYSTEM_HEADER_START + DEFAULT_SYSTEM_PROMPT + EOS_TOKEN

    # 이후 메시지들 처리
    for message in messages:
        if message["role"] == "user":
            formatted_text += USER_HEADER_START + message["content"] + EOS_TOKEN
        elif message["role"] == "assistant":
            formatted_text += ASSISTANT_HEADER_START + message["content"].strip() + EOS_TOKEN
        else:
            raise ValueError(
                "LLaMA 3 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
            )

    # 마지막 assistant 응답 시작 토큰 삽입 (모델이 이어서 생성하게 하기 위함)
    formatted_text += ASSISTANT_HEADER_START

    # BOS 토큰 추가 여부
    if add_bos:
        formatted_text = BOS_TOKEN + formatted_text

    return formatted_text

def convert_qa_to_llama3_prompts(qa_list):
    return [
        create_prompt_with_llama3_chat_format([
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ])
        for example in qa_list
    ]
'''
# 예시 QA 샘플 1개
qa_dataset = [
    {
        "question": "Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?",
        "answer": "The author in question is Jaime Vasquez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre."
    },
    {
        "question": "What is the capital of Italy?",
        "answer": "The capital of Italy is Rome."
    }
]

# 프롬프트 변환
prompts = convert_qa_to_llama3_prompts(qa_dataset)

# 결과 출력
for i, prompt in enumerate(prompts):
    print(f"--- Prompt {i+1} ---")
    print(prompt)
    print()

### answer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 로드 (LLaMA 3 기반)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # 예시
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# 질문
question = "What is the capital of South Korea?"

# 프롬프트 생성
prompt = convert_question_to_inference_prompt(question)

# 토큰화
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 생성
output = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")  # 종료 토큰 설정
)

# 결과 디코딩
response = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print(response.strip())


'''