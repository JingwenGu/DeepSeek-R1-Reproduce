# region imports
from transformers import HfArgumentParser
import os
import json
from dataclasses import dataclass, field
from typing import Optional
#endregion

# region args
@dataclass
class ScriptArguments:
    input_dir: str = field(default="", metadata={"help": "# input dir"})
    output_dir: str = field(default="", metadata={"help": "# output dir"})
    threshold_correct: Optional[float] = field(default="", metadata={"help": "minimum reward for correctness"})
    threshold_strict: Optional[float] = field(default="", metadata={"help": "minimum reward for strict format"})
    threshold_soft: Optional[float] = field(default="", metadata={"help": "minimum reward for soft format"})
    threshold_int: Optional[float] = field(default="", metadata={"help": "minimum reward for int result"})
    threshold_xml: Optional[float] = field(default="", metadata={"help": "minimum reward for xml count"})
    comments: Optional[str] = field(default="", metadata={"help": "comments"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
# endregion

with open(script_args.input_dir,'r') as file:
    inference_data = json.load(file)

selected = []

# for problem in inference_data:
#     for completion,r_correct,r_strict,r_soft,r_int,r_xml in zip(problem['completion'],problem['correctness_reward'],problem['strict_format_reward'],problem['soft_format_reward'],problem['int_reward'],problem['xmlcount_reward']):
#         # print(r_correct,r_strict,r_soft,r_int,r_xml)
#         if r_correct >= script_args.threshold_correct and r_strict >= script_args.threshold_strict and r_soft >= script_args.threshold_soft and r_int >= script_args.threshold_int and r_xml >= script_args.threshold_xml:
#             selected.append({'question':problem['question'],'completion':completion})

for problem in inference_data:
    for completion,r_correct,r_strict,r_soft,r_xml in zip(problem['completion'],problem['correctness_reward'],problem['strict_format_reward'],problem['soft_format_reward'],problem['xmlcount_reward']):
        # print(r_correct,r_strict,r_soft,r_int,r_xml)
        if r_correct >= script_args.threshold_correct and r_strict >= script_args.threshold_strict and r_soft >= script_args.threshold_soft and r_xml >= script_args.threshold_xml:
            selected.append({'question':problem['question'],'completion':completion})

with open(script_args.output_dir,'w') as file:
    json.dump(selected,file)

print(f'Selected {len(selected)} samples')