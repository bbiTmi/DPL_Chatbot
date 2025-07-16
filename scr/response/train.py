import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "response_data.csv")

dataset = load_dataset("csv", data_files=DATA_PATH)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

model_checkpoint = "vinai/bartpho-syllable"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

def preprocess(example):
    inputs = tokenizer(
        example["input"],
        max_length=32,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        example["target"],
        max_length=64,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="model/bartpho",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("model/bartpho")
    tokenizer.save_pretrained("model/bartpho")
