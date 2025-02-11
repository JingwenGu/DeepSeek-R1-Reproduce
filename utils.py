import json
import torch
import random
from datasets import Dataset,concatenate_datasets
from transformers import DataCollatorForLanguageModeling
#from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

############################__COLLATOR+DATALOADER__#################################

# Tokenizes 1 single column from dataset
# Requires untokenized dataset + tokenizer
def getTokenizedDataLoader(dataset,tokenizer,column,batch_size,max_length,shuffle=False):
    def tokenize_function(examples):
        return tokenizer(examples[column], return_tensors='pt', padding="max_length", truncation=True, max_length=max_length) # used to be 128
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        # remove_columns=[column]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    dataloader = torch.utils.data.DataLoader(
        tokenized_datasets, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )
    return dataloader

# Requires tokenized dataset + tokenizer
def getDataLoader(tokenized_datasets,tokenizer,batch_size,shuffle=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    dataloader = torch.utils.data.DataLoader(
        tokenized_datasets, 
        batch_size=batch_size,  # Adjust batch size based on your GPU memory
        shuffle=shuffle,
        collate_fn=data_collator
    )
    return dataloader