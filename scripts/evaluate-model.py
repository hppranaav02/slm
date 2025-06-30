import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
)
from sklearn.metrics import classification_report, accuracy_score

# --- Config ---
MODEL_DIR = "../models/codet5p-go-cwe/checkpoint-4832"
DATA_FILE = "../data/check_for_2015.jsonl"
OUTPUT_REPORT = "../outputs/eval_report.json"

# --- Load model + tokenizer ---
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# --- Load and split dataset ---
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.train_test_split(test_size=0.2)
test_data = dataset["test"]

# --- Tokenize test set ---
def preprocess(examples):
    inputs = [f"{i} {c}" for i, c in zip(examples["instruction"], examples["input"])]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=64)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_test = test_data.map(preprocess, batched=True)
collator = DataCollatorForSeq2Seq(tokenizer, model=model)
training_args = Seq2SeqTrainingArguments(
    output_dir="../outputs/test-run",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    do_train=False,
    do_eval=False,
    logging_dir="../outputs/logs"
)

# --- Inference ---
trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer,args=training_args, data_collator=collator)
raw_preds = trainer.predict(tokenized_test)

# --- Decode predictions ---
pred_tokens = tokenizer.batch_decode(raw_preds.predictions, skip_special_tokens=True)
true_labels = [x["output"].strip() for x in test_data]
pred_labels = [x.strip() for x in pred_tokens]

# --- Metrics ---
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

# --- Save full report to JSON ---
os.makedirs("outputs", exist_ok=True)
with open(OUTPUT_REPORT, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "classification_report": report
    }, f, indent=2)

# --- Print key metrics ---
print(f"Accuracy: {accuracy:.4f}")
print("Top-level F1 scores:")
for label, scores in report.items():
    if isinstance(scores, dict) and "f1-score" in scores:
        print(f"{label:>30}: F1 = {scores['f1-score']:.4f}")
