import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

# --- Configuration ---
MODEL_NAME = "Salesforce/codet5p-220m"
DATA_PATH = "../data/check_for_2015.jsonl"
OUTPUT_DIR = "models/codet5p-go-cwe"

# --- Load and split dataset ---
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.2)

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Preprocess function ---
def preprocess(examples):
    inputs = [f"{i} {c}" for i, c in zip(examples["instruction"], examples["input"])]
    model_inputs = tokenizer(inputs, truncation=True, max_length=512, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], truncation=True, max_length=64, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

# --- Load model ---
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --- Define training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=4,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,  # Safe for RTX 4060
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    save_total_limit=2
)

# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer
)

# --- Train! ---
trainer.train()
