import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=4):
    data_files = [
        "../data/000_00000.parquet",
        "../data/000_00001.parquet",
        "../data/000_00002.parquet",
        "../data/000_00003.parquet"
    ]
    
    full_dataset = load_dataset("parquet", data_files=data_files, split="train")
    
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    
    dataset_train = split_dataset['train']
    dataset_val = split_dataset['test']


    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tokenized_datasets = dataset_train.map(tokenize_function, batched=True, num_proc=4)
    tokenized_val = dataset_val.map(tokenize_function, batched=True, num_proc=4)

    train_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    return train_loader, val_loader, tokenizer
