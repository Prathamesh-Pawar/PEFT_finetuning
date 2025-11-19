import os
import torch
from datasets import load_dataset,load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

# -----------------------------------------------------------------------------
# 1. Environment setup
# -----------------------------------------------------------------------------

os.environ["WANDB_API_KEY"] = "423405722831be2e96dd472655828b1ef9978bc9"

from huggingface_hub import login
login(new_session=False)

# Optional bitsandbytes optimizer
try:
    import bitsandbytes as bnb
    AdamW8bit = bnb.optim.AdamW8bit
except Exception:
    AdamW8bit = None

# -----------------------------------------------------------------------------
# 2. Model + tokenizer setup
# -----------------------------------------------------------------------------

base_model_name = "meta-llama/Llama-3.2-1B"
dataset_name = "11_18_25_100k_English_Marathi"
run_name = "11_18_25_100k_English_Marathi"

os.mkdir("model_runs/" + run_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# -----------------------------------------------------------------------------
# 3. Configure LoRA
# -----------------------------------------------------------------------------

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# -----------------------------------------------------------------------------
# 4. Load tokenized dataset
# -----------------------------------------------------------------------------

tokenized = load_from_disk("training_datasets/"+dataset_name)
print(tokenized)

# -----------------------------------------------------------------------------
# 5. Data collator
# -----------------------------------------------------------------------------

def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
    attention_mask = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# -----------------------------------------------------------------------------
# 6. TrainingArguments
# -----------------------------------------------------------------------------

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    eval_strategy="epoch" if "validation" in tokenized else "no",
    save_strategy="epoch",
    output_dir="output/",
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# -----------------------------------------------------------------------------
# 7. Optimizer
# -----------------------------------------------------------------------------

optim = None
if AdamW8bit is not None:
    print("Using bitsandbytes AdamW8bit optimizer")
    optim = AdamW8bit(model.parameters(), lr=2e-4)

# -----------------------------------------------------------------------------
# 8. Trainer
# -----------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"] if "test" in tokenized else None,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    optimizers=(optim, None) if optim is not None else (None, None),
)

# -----------------------------------------------------------------------------
# 9. Train
# -----------------------------------------------------------------------------

trainer.train()

# -----------------------------------------------------------------------------
# 10. Save LoRA adapter
# -----------------------------------------------------------------------------

print("Saving PEFT (LoRA) weights...")
model.save_pretrained(os.path.join("model_runs/" + run_name, "peft_lora"))

print("Done.")
