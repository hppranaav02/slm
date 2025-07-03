import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from data_model import ResponseFormat, instruction

# --- Configuration ---
MODEL_NAME = "Salesforce/codet5p-220m"
DATA_PATH = "./check_for_2015.jsonl"
OUTPUT_DIR = "models/codet5p-go-cwe"

# --- Load and split dataset ---
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.2)
print(f"Dataset loaded with {len(dataset['train'])} training examples and {len(dataset['test'])} test examples.")

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded successfully.")

# --- Preprocess function ---
def preprocess(examples):
    inputs = [f"instruction:\n{i}\ninput:{c}" for i, c in zip([instruction]*len(examples["input"]), examples["input"])]
    model_inputs = tokenizer(inputs, truncation=True, max_length=512, padding="max_length")
    print("Inputs tokenized successfully.")

    # Pad the tokens to maximum length

    #ask what this does
    with tokenizer.as_target_tokenizer():
        outputs= []
        for output in zip(examples["output"]):
            print(f"Processing output: {output[0]}")
            vul, vul_type = output[0].split("=") if output[0].lower() != 'secure' else (output[0], None)
            vul = vul.strip()
            vul_type = vul_type.strip() if vul_type else None
            response = ResponseFormat(
                type="json",
                vulnerability=(vul.lower() == "vulnerable"),
                vulnerability_type=vul_type,
            )
            # examples["output"] = f"```json\n{response.json()}\n```"
            outputs.append((f"```json\n{response.json()}\n```"))
        examples["output"] = outputs

        labels = tokenizer(examples["output"], truncation=True, max_length=128, padding="max_length") # try 128,

    model_inputs["labels"] = labels["input_ids"]
    print("Labels tokenized successfully.")
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)
print("Dataset tokenized successfully.")

# --- Load model ---
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")

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
print("Training arguments defined successfully.")

# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer
)
print("Trainer initialized successfully.")

# --- Train! ---
trainer.train()
print("Training started...")
