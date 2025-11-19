import os
import random
from datasets import load_dataset

from transformers import AutoTokenizer


from huggingface_hub import login

login(token="API_KEY")
# ----------------------------------------------------------
# 1. Load tokenizer + dataset
# ----------------------------------------------------------

base_model_name = "meta-llama/Llama-3.2-1B"
dataset_name = "anujsahani01/English-Marathi"
run_name = "11_18_25_100k_English_Marathi"
os.mkdir('training_datasets/'+run_name)
output_file = 'training_datasets/'+run_name

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load 30% of dataset
dataset = load_dataset(dataset_name, split="train[:30%]")

# ----------------------------------------------------------
# 2. Create random subset
# ----------------------------------------------------------
indices = random.sample(range(len(dataset)), 105000)

dataset = dataset.select(indices)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset['train']
test_dataset = dataset['test']


train_dataset = train_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = train_dataset.rename_columns({
    "english": "source",
    "marathi": "target"
})

test_dataset = test_dataset.rename_columns({
    "english": "source",
    "marathi": "target"
})



# ----------------------------------------------------------
# 3. Prompt builder
# ----------------------------------------------------------

def build_prompt(source_text: str, source_lang: str, target_lang: str):
    return (
        f"Translate from {source_lang} to {target_lang}:\n\n"
        f"{source_text}\n\n"
        f"Translation:"
    )

# ----------------------------------------------------------
# 4. Tokenization + chunking
# ----------------------------------------------------------

def tokenize_and_chunk(examples, tokenizer, max_length, source_lang, target_lang):
    inputs = []
    labels = []

    for src, tgt in zip(examples["source"], examples["target"]):
        prompt = build_prompt(src, source_lang, target_lang)
        full = prompt + " " + tgt

        tokenized_full = tokenizer(
            full,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        # Compute length of prompt without padding
        prompt_len = len(tokenizer(prompt)["input_ids"])

        # Initialize labels
        labels_ids = [-100] * len(input_ids)
        full_ids = tokenized_full["input_ids"]

        # Target token region starts after prompt
        start_idx = prompt_len
        for i in range(start_idx, len(full_ids)):
            labels_ids[i] = full_ids[i]

        inputs.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
        })

    return {
        "input_ids": [x["input_ids"] for x in inputs],
        "attention_mask": [x["attention_mask"] for x in inputs],
        "labels": [x["labels"] for x in inputs],
    }

# ----------------------------------------------------------
# 6. Preprocess dataset
# ----------------------------------------------------------

tokenizer.pad_token = tokenizer.eos_token

def preprocess_fn(batch):
    return tokenize_and_chunk(batch, tokenizer, 512, "English", "Marathi")

tokenized = train_dataset.map(
    lambda examples: preprocess_fn(examples),
    batched=True,
    remove_columns=train_dataset["train"].column_names,
)

tokenized.save_to_disk(output_file)
test_dataset.save_to_disk("evaluation_dataset/"+run_name)