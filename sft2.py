# region imports
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, DataCollatorForLanguageModeling, Trainer, TrainingArguments
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
    model_name: str = field() # ckpt/0208/rl0_Qwen-05B/checkpoint-400
    tokenizer_name: str = field() # Qwen/Qwen2.5-0.5B-Instruct
    rollout_dir: Optional[str] = field(default="test", metadata={"help": "the dataset split"})
    num_epochs: Optional[int] = field(default=3, metadata={"help": "the number of epochs"}) # 3
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"}) # 16
    save_steps: Optional[int] = field(default=100, metadata={"help": "# steps to save the model"}) # 100
    eval_steps: Optional[int] = field(default=100, metadata={"help": "# steps to eval the model"}) # 10
    log_steps: Optional[int] = field(default=10, metadata={"help": "# steps to log the model"}) # 10
    max_prompt_length: Optional[int] = field(default=200, metadata={"help": "max prompt length"}) # 256
    max_completion_length: Optional[int] = field(default=200, metadata={"help": "max completion length"}) # 200
    log_dir: str = field(default="", metadata={"help": "log dir"}) # .ckpt/0209/sft1_Qwen-05B/logs
    output_dir: str = field(default="", metadata={"help": "output dir"}) # .ckpt/0209/sft1_Qwen-05B
    run_name: str = field(default="eval_res", metadata={"help": "wandb run name"}) # ckpt/0208/rl0_Qwen-05B
    comments: Optional[str] = field(default="", metadata={"help": "comments"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

os.makedirs(script_args.output_dir, exist_ok=True)

args_dict = {'model_name':script_args.model_name,
            'tokenizer_name':script_args.tokenizer_name,
            'rollout_dir':script_args.rollout_dir,
            'num_epochs':script_args.num_epochs,
            'batch_size':script_args.batch_size,
            'save_steps':script_args.save_steps,
            'eval_steps':script_args.eval_steps,
            'log_steps':script_args.log_steps,
            'max_prompt_length':script_args.max_prompt_length,
            'max_completion_length':script_args.max_completion_length,
            'log_dir':script_args.log_dir,
            'output_dir':script_args.output_dir,
            'run_name':script_args.run_name,
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

# region model + tokenizer + dataset
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

with open(script_args.rollout_dir,'r') as file:
    rollout_samples = json.load(file)

def convert_sample(sample):
    prompt = [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': sample['question']}]
    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt_text + sample['completion'] + tokenizer.eos_token

converted_samples = [convert_sample(sample) for sample in rollout_samples]

# print(converted_samples[:10])

dataset = Dataset.from_dict({'text':converted_samples}).train_test_split(test_size=0.05)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=script_args.max_prompt_length+script_args.max_completion_length)
tokenized_dataset_train = dataset['train'].map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset_test = dataset['test'].map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# endregion

# region Training arguments
wandb.init(
    project="DeepSeek_SFT1",
    name=script_args.run_name
    # config={"epochs": 3, "batch_size": 8}
)

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_steps,
    save_strategy="steps",
    save_steps=script_args.save_steps,
    per_device_train_batch_size=script_args.batch_size,
    per_device_eval_batch_size=script_args.batch_size,
    num_train_epochs=script_args.num_epochs,
    logging_dir=script_args.log_dir,
    logging_steps=script_args.log_steps,
    report_to="wandb", 
    save_total_limit=None,
    # fp16=True,
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    tokenizer=tokenizer,
    data_collator=data_collator
)
# endregion

trainer.train()