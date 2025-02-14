# region imports
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import wandb
import datetime
import os
import json
from dataclasses import dataclass, field
from typing import Optional
# from utils import getTokenizedDataLoader
#endregion

# region args
@dataclass
class ScriptArguments:
    model_name: str = field() # Qwen/Qwen2.5-0.5B-Instruct
    tokenizer_name: str = field() # Qwen/Qwen2.5-0.5B-Instruct
    # dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    # dataset_split: Optional[str] = field(default="test", metadata={"help": "the dataset split"})
    ckpt: Optional[str] = field(default="", metadata={"help": "ckpts to eval, separated by underlines, e.g. 10_20_30"})
    # test_size: Optional[int] = field(default=16, metadata={"help": "the number of test samples for each ckpt"}) # 16
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"}) # 16
    num_generations: Optional[int] = field(default=200, metadata={"help": "number of generations"}) # 16
    max_prompt_length: Optional[int] = field(default=200, metadata={"help": "max prompt length"}) # 256
    max_completion_length: Optional[int] = field(default=200, metadata={"help": "max completion length"}) # 200
    output_dir: str = field(default="", metadata={"help": "output dir"}) # ckpt/0208/rl0_Qwen-05B
    output_name: str = field(default="eval_res", metadata={"help": "name of output file"}) # ckpt/0208/rl0_Qwen-05B
    comments: Optional[str] = field(default="", metadata={"help": "comments"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

# os.makedirs(script_args.output_dir, exist_ok=True)

args_dict = {'model_name':script_args.model_name,
            'ckpt':script_args.ckpt,
            'batch_size':script_args.batch_size,
            'num_generations':script_args.num_generations,
            'max_prompt_length':script_args.max_prompt_length,
            'max_completion_length':script_args.max_completion_length,
            'output_dir':script_args.output_dir,
            'comments':script_args.comments}
with open(f'{script_args.output_dir}/args.json', 'w') as file:
    json.dump(args_dict,file)
print(args_dict)
# endregion

# region prompt templates
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
# endregion

# region tokenizer + GSM8K dataset
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train", num_samples = -1) -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.shuffle()
    if num_samples > 0:
        data = data.select(range(num_samples))
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def get_math500_questions() -> Dataset:
    data = load_dataset('di-zhang-fdu/MATH500')['test']
    data = data.shuffle()
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer']
    }) # type: ignore
    return data # type: ignore

dataset = get_math500_questions()
dataset = Dataset.from_dict(dataset[400:]) # last 100 from MATH500 for testing
# endregion

# region dataloader (doesn't work)
def template_func(example):
    example['prompt_text'] = tokenizer.apply_chat_template(
        example['prompt'],
        tokenize=False,
        add_generation_prompt=True
    )
    return example
dataset_test = dataset.map(
    template_func, 
    batched=True, 
    remove_columns=['prompt']
)

def getTokenizedDataLoader(dataset,tokenizer,column,batch_size,max_length,shuffle=False):
    def tokenize_function(examples):
        return tokenizer(examples[column], return_tensors='pt', padding=True, truncation=True, max_length=max_length) # used to be 128
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=[column]
    )
    def data_collator(batch):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)  # Ensure padding
        questions = [example["question"] for example in batch]
        answers = [example["answer"] for example in batch]
        return {"input_ids": input_ids, "question": questions, "answer":answers}

    dataloader = torch.utils.data.DataLoader(
        tokenized_datasets, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )
    return dataloader
dataloader = getTokenizedDataLoader(dataset=dataset_test,
                                    tokenizer=tokenizer,
                                    column='prompt_text',
                                    batch_size=1,
                                    max_length=256,
                                    shuffle=True)
# endregion

# region Reward functions
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # print('='*10 + "completions:" + '='*10)
    # print(type(completions))
    # print(completions)
    # print('='*10 + "completions[0]:" + '='*10)
    # print(type(completions[0]))
    # print(completions[0])
    # print('='*10 + "prompts" + '='*10)
    # print(type(prompts))
    # print(prompts)
    # print('='*10 + "answer" + '='*10)
    # print(type(answer))
    # print(answer)
    extracted_responses = [extract_xml_answer(r) for r in completions]
    print('-'*20, f"Question:\n{prompts}", f"\nAnswer:\n{answer}", f"\nResponse:\n{completions[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == answer else 0.0 for r in extracted_responses]

def int_reward_func(completions, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(r) for r in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]
#endregion

# region inference for each model
def inference0(model):
    results = []
    model.eval()
    device = model.device
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            input_ids.to(device)
            # attention_mask = batch['attention_mask'].to(device)

            print('='*10 + 'input_ids' + '='*10)
            print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

            outputs = model.generate(
                input_ids, 
                max_new_tokens=script_args.max_prompt_length + script_args.max_completion_length, 
                # num_return_sequences=script_args.num_generations,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                # repetition_penalty=2.0,
                # length_penalty=1.2,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            print('='*10 + 'outputs' + '='*10)
            decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(decoded_texts)

            def reward_calc(func,completions):
                rewards = func(completions)
                avg_reward = sum(rewards)/len(rewards)
                return rewards,avg_reward

            prompt = batch['question']
            answer = batch['answer']
            correctness_reward = correctness_reward_func(prompt, decoded_texts, answer)
            avg_correctness_reward = sum(correctness_reward)/len(correctness_reward)

            int_reward, avg_int_reward = reward_calc(int_reward_func, decoded_texts)
            strict_format_reward, avg_strict_format_reward = reward_calc(strict_format_reward_func, decoded_texts)
            soft_format_reward, avg_soft_format_reward = reward_calc(soft_format_reward_func, decoded_texts)
            xmlcount_reward, avg_xmlcount_reward = reward_calc(xmlcount_reward_func, decoded_texts)

            total_reward = [r0+r1+r2+r3+r4 for r0,r1,r2,r3,r4 in zip(correctness_reward,int_reward,strict_format_reward,soft_format_reward,xmlcount_reward)]
            avg_total_reward = avg_correctness_reward + avg_int_reward + avg_strict_format_reward + avg_soft_format_reward + avg_xmlcount_reward

            results.append({'prompt':prompt,
                            'answer':answer,
                            'completion':decoded_texts,
                            'correctness_reward':correctness_reward,
                            'int_reward':int_reward,
                            'strict_format_reward':strict_format_reward,
                            'soft_format_reward':soft_format_reward,
                            'xmlcount_reward':xmlcount_reward,
                            'total_reward':total_reward,
                            'avg_correctness_reward':avg_correctness_reward,
                            'avg_int_reward':avg_int_reward,
                            'avg_strict_format_reward':avg_strict_format_reward,
                            'avg_soft_format_reward':avg_soft_format_reward,
                            'avg_xmlcount_reward':avg_xmlcount_reward,
                            'avg_total_reward':avg_total_reward})
    return results

def truncate(response):
    if response.find('</answer>\n') == -1:
        return response
    return response[:response.find('</answer>')+10]

def inference(model):
    results = []
    model.eval()
    device = model.device
    with torch.no_grad():
        for question, answer, prompt in zip(dataset['problem'],dataset['answer'],dataset['prompt']):
            prompt_text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

            outputs = model.generate(
                input_ids, 
                max_new_tokens=script_args.max_completion_length, 
                num_return_sequences=script_args.num_generations,
                # num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                # repetition_penalty=2.0,
                # length_penalty=1.2,
                # temperature=0.7,
                # top_p=0.9,
                # do_sample=True,
            )
            decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            completion = [response[response.find('assistant\n')+10:] for response in decoded_texts]
            completion = [truncate(response) for response in completion]

            def reward_calc(func,completions):
                rewards = func(completions)
                avg_reward = sum(rewards)/len(rewards)
                return rewards,avg_reward

            correctness_reward = correctness_reward_func(question, completion, answer)
            avg_correctness_reward = sum(correctness_reward)/len(correctness_reward)

            # int_reward, avg_int_reward = reward_calc(int_reward_func, completion)
            strict_format_reward, avg_strict_format_reward = reward_calc(strict_format_reward_func, completion)
            soft_format_reward, avg_soft_format_reward = reward_calc(soft_format_reward_func, completion)
            xmlcount_reward, avg_xmlcount_reward = reward_calc(xmlcount_reward_func, completion)

            total_reward = [r0+r1+r2+r3 for r0,r1,r2,r3 in zip(correctness_reward,strict_format_reward,soft_format_reward,xmlcount_reward)]
            avg_total_reward = avg_correctness_reward + avg_strict_format_reward + avg_soft_format_reward + avg_xmlcount_reward

            results.append({'question':question,
                            'answer':answer,
                            'completion':completion,
                            'correctness_reward':correctness_reward,
                            # 'int_reward':int_reward,
                            'strict_format_reward':strict_format_reward,
                            'soft_format_reward':soft_format_reward,
                            'xmlcount_reward':xmlcount_reward,
                            'total_reward':total_reward,
                            'avg_correctness_reward':avg_correctness_reward,
                            # 'avg_int_reward':avg_int_reward,
                            'avg_strict_format_reward':avg_strict_format_reward,
                            'avg_soft_format_reward':avg_soft_format_reward,
                            'avg_xmlcount_reward':avg_xmlcount_reward,
                            'avg_total_reward':avg_total_reward})
    return results
# endregion

if len(script_args.ckpt) > 0:
    ckpts = script_args.ckpt.split('_')
    ckpts = [int(c) for c in ckpts]
    models = [f'{script_args.model_name}/checkpoint-{c}' for c in ckpts]
else:
    models = [script_args.model_name]

for model_name in models:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    results = inference(model)
    keyword = "eval" if script_args.dataset_split == 'test' else 'sample'
    with open(f'{model_name}/{script_args.output_name}.json','w') as file:
        json.dump(results,file)