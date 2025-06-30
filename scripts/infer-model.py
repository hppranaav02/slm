from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Load model and tokenizer ---
MODEL_PATH = "../models/codet5p-go-cwe/checkpoint-4832"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# --- Inference pipeline ---
classifier = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# --- Inference function ---
def classify_go_code(code_snippet):
    instruction = "Classify the Go code as Vulnerable or Secure with CWE ID."
    prompt = f"{instruction} {code_snippet}"
    result = classifier(prompt, max_new_tokens=32, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"]

# --- Example usage ---
if __name__ == "__main__":
    go_code = '''
    package main

    import "fmt"

    func main() {
        userInput := "<script>alert(1)</script>"
        fmt.Printf(userInput)
    }
    '''
    result = classify_go_code(go_code)
    print("Classification Result:", result)
