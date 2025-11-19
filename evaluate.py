import os
import math
import json
import torch
from tqdm import tqdm
from evaluate import load
from datasets import load_from_disk
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from comet import download_model, load_from_checkpoint


# ------------------------------------------------------
# 1. Login to HuggingFace
# ------------------------------------------------------
def hf_login(token: str):
    login(token=token)


# ------------------------------------------------------
# 2. Load Model + LoRA Adapter
# ------------------------------------------------------
def load_translation_model(base_model_name: str, adapter_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)

    return tokenizer, model, device


# ------------------------------------------------------
# 3. Load Dataset
# ------------------------------------------------------
def load_dataset_from_disk_path(path: str):
    return load_from_disk(path)


# ------------------------------------------------------
# 4. Translation Function
# ------------------------------------------------------
def translate(
        text: str,
        tokenizer,
        model,
        device,
        source_lang="English",
        target_lang="Marathi",
        max_new_tokens=128,
):
    prompt = (
        f"Translate the following {source_lang} sentence to {target_lang}:\n"
        f"{source_lang}: {text}\n{target_lang}:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if f"{target_lang}:" in translated:
        translated = translated.split(f"{target_lang}:")[-1].strip()

    return translated


# ------------------------------------------------------
# 5. Compute Evaluation Metrics
# ------------------------------------------------------
def evaluate_metrics(src_texts, pred_texts, ref_texts, comet_model):
    # CHRF
    chrf = load("chrf")
    chrf_score = chrf.compute(predictions=pred_texts, references=ref_texts)
    print("CHRF:", chrf_score)

    # BLEU
    bleu = load("bleu")
    bleu_score = bleu.compute(predictions=pred_texts, references=ref_texts)
    print("BLEU:", bleu_score)

    # COMET
    comet_inputs = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(src_texts, pred_texts, ref_texts)
    ]
    comet_scores = comet_model.predict(comet_inputs)
    print("COMET:", comet_scores.system_score)

    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "comet": comet_scores.system_score,
    }


# ------------------------------------------------------
# 6. Perplexity
# ------------------------------------------------------
def compute_perplexity(model, tokenizer, texts, device):
    ppl_evaluator = load("perplexity")

    results = ppl_evaluator.compute(
        model_id=None,
        predictions=texts,
        tokenizer=tokenizer,
        model=model,
        device=str(device)
    )

    print("Perplexity:", results)
    return results


# ------------------------------------------------------
# 7. Save Metrics
# ------------------------------------------------------
def save_metrics(metrics, output_path="metrics.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_path}")


# ------------------------------------------------------
# 8. Main Execution
# ------------------------------------------------------
if __name__ == "__main__":
    # >>> Replace with your real token
    hf_login(token="API_KEY")

    # Paths and model names
    base_model_name = "meta-llama/Llama-3.2-1B"
    dataset_name = "11_18_25_100k_English_Marathi"
    run_name = "11_18_25_100k_English_Marathi"

    # Load dataset
    dataset = load_dataset_from_disk_path(f"evaluation_dataset/{dataset_name}")

    # Load translation model + LoRA
    tokenizer, model, device = load_translation_model(
        base_model_name,
        adapter_path=f"model_runs/{run_name}"
    )

    # Load COMET model
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    # Extract test data
    src_texts = dataset["test"]["source"]
    ref_texts = dataset["test"]["target"]

    # Run translations
    pred_texts = [
        translate(text, tokenizer, model, device)
        for text in tqdm(src_texts, desc="Translating", unit="sentence")
    ]

    # Evaluate all translation metrics
    metrics = evaluate_metrics(src_texts, pred_texts, ref_texts, comet_model)

    # Compute perplexity
    ppl_results = compute_perplexity(model, tokenizer, ref_texts, device)

    # Add it to the metrics dictionary
    metrics["perplexity"] = ppl_results

    # Save to disk
    save_metrics(
        metrics,
        output_path=f"metrics/{run_name}_results.json"
    )
