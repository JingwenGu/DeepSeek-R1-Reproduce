# region imports
"""
References:
https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
https://github.com/waylandzhang/DeepSeek-RL-Qwen-0.5B-GRPO-gsm8k/blob/main/train-checkpoint-900.ipynb
"""
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
#endregion

# region args
@dataclass
class ScriptArguments:
    model_name: str = field() # Qwen/Qwen2.5-0.5B-Instruct
    # dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    num_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"}) # 1
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"}) # 16
    num_generations: Optional[int] = field(default=200, metadata={"help": "number of generations"}) # 16
    max_prompt_length: Optional[int] = field(default=200, metadata={"help": "max prompt length"}) # 256
    max_completion_length: Optional[int] = field(default=200, metadata={"help": "max completion length"}) # 200
    grad_steps: Optional[int] = field(default=10, metadata={"help": "# steps to update gradient"}) # 4 
    save_steps: Optional[int] = field(default=50, metadata={"help": "# steps to save the model"}) # 10
    log_steps: Optional[int] = field(default=1, metadata={"help": "# steps to log the model"}) # 1
    lr: Optional[float] = field(default=5e-6, metadata={"help": "# learning rate"}) # 5e-6
    lr_type: Optional[str] = field(default="cosine", metadata={"help": "# lr scheduler type"}) # cosine
    output_dir: str = field(default="", metadata={"help": "# output dir"}) # ckpt/0208/rl0_Qwen-05B
    wandb_name: Optional[str] = field(default="", metadata={"help": "wandb project name"}) # rl0_Qwen-05B_0208
    comments: Optional[str] = field(default="", metadata={"help": "comments"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

os.makedirs(script_args.output_dir, exist_ok=True)

args_dict = {'model_name':script_args.model_name,
            'num_epochs':script_args.num_epochs,
            'batch_size':script_args.batch_size,
            'num_generations':script_args.num_generations,
            'max_prompt_length':script_args.max_prompt_length,
            'max_completion_length':script_args.max_completion_length,
            'grad_steps':script_args.grad_steps,
            'save_steps':script_args.save_steps,
            'log_steps':script_args.log_steps,
            'lr':script_args.lr,
            'lr_type':script_args.lr_type,
            'output_dir':script_args.output_dir,
            'wandb_name':script_args.wandb_name,
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

# region GSM8K dataset
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()
# endregion

# region Reward functions
def truncate(response):
    if response.find('</answer>\n') == -1:
        return response
    return response[:response.find('</answer>')+10]

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    responses = [truncate(response) for response in responses]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    responses = [truncate(response) for response in responses]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    responses = [truncate(response) for response in responses]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    responses = [truncate(response) for response in responses]
    matches = [re.match(pattern, r) for r in responses]
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
    contents = [completion[0]["content"] for completion in completions]
    contents = [truncate(response) for response in contents]
    return [count_xml(c) for c in contents]
#endregion

# region load model
model_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=None
# ).to("cuda")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
# endregion

# region GRPO args
output_dir=script_args.output_dir
run_name=script_args.wandb_name

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=script_args.lr,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type=script_args.lr_type,
    logging_steps=script_args.log_steps,
    bf16=True,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.grad_steps,
    num_generations=script_args.num_generations,
    max_prompt_length=script_args.max_prompt_length,
    max_completion_length=script_args.max_completion_length,
    num_train_epochs=script_args.num_epochs,
    save_steps=script_args.save_steps,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to=["wandb"]
)

class LogOutputsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        outputs = kwargs.get("outputs", {})
        with open(f"{output_dir}/train_outputs.txt", "a") as f:
            f.write(f"Step {state.global_step}: {outputs}\n\n")

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    callbacks=[LogOutputsCallback()],
    #peft_config=peft_config
)
# endregion

trainer.train()

trainer.save_model(output_dir)