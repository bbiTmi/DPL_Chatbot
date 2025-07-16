import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Đường dẫn tới mô hình đã huấn luyện
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "bartpho"))

# Load model và tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

def generate_response(user_input: str, intent: str = None):
    if intent:
        prompt = f"<intent>{intent}</intent> {user_input}"
    else:
        prompt = user_input
        
    prompt = user_input  # hoặc f"[{intent}] {user_input}"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=32)

    # Sinh output
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
