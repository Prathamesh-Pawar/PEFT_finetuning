from huggingface_hub import login
login(new_session=False)

from tqdm import tqdm
import pandas as pd
from evaluate import load
from sacrebleu import corpus_bleu, corpus_chrf

from comet import download_model, load_from_checkpoint
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

import random

import torch
from training_datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig

# Optional: use 8-bit Adam from bitsandbytes if installed
try:
    import bitsandbytes as bnb
    AdamW8bit = bnb.optim.AdamW8bit
except Exception:
    AdamW8bit = None
torch.cuda.empty_cache()


dataset = load_dataset("anujsahani01/English-Marathi",split="train[:30%]")


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",  # or "cuda" if single GPU
    torch_dtype="auto"
)


from peft import PeftModel

# 3. Load LoRA adapter on top of the base model
model = PeftModel.from_pretrained(base_model, "output_100k/peft_lora")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Move model to GPU
base_model.to(device)


def translate_text(text: str, source_lang="English", target_lang="Marathi", max_new_tokens=128):
    """
    Generate a translation using the fine-tuned LoRA model.
    """
    prompt = f"Translate the following {source_lang} sentence to {target_lang}:\n{source_lang}: {text}\n{target_lang}:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the translation part (after 'Marathi:')
    if f"{target_lang}:" in translated:
        translated = translated.split(f"{target_lang}:")[-1].strip()
    return translated

src_texts = subset['test']['source'][:]
ref_texts = subset['test']['target'][:]


pred_texts = []

for text in tqdm(src_texts, desc="Translating", unit="sentence"):
    translation = translate_text(text)
    pred_texts.append(translation)