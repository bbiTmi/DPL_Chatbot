import tensorflow as tf
import numpy as np
import joblib
import sqlite3
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from word2number import w2n
from keras.layers import TFSMLayer

# MODEL_PATH = "model/best_model_formatted"
# model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "coffee.db")
SLOT_MODEL_PATH = os.path.join(BASE_DIR, "model", "slot_model.keras")
WORD2IDX_PATH = os.path.join(BASE_DIR, "model", "word2idx.pkl")
IDX2TAG_PATH = os.path.join(BASE_DIR, "model", "idx2tag.pkl")
MAX_LEN = 30

slot_model = load_model(SLOT_MODEL_PATH)
word2idx = joblib.load(WORD2IDX_PATH)
idx2tag = joblib.load(IDX2TAG_PATH)

def predict_slots(text):
    tokens = text.strip().split()
    x = [word2idx.get(w, word2idx["UNK"]) for w in tokens]
    x = pad_sequences([x], maxlen=MAX_LEN, padding='post')
    x = tf.constant(x, dtype=tf.float32)

    y_pred = slot_model(x)  # Tensor shape: (1, max_len, num_tags)
    y_pred = y_pred.numpy()[0]
    pred_ids = np.argmax(y_pred, axis=-1)
    slot_tags = [idx2tag[i] for i in pred_ids[:len(tokens)]]
    return list(zip(tokens, slot_tags))


def extract_entities(token_slot_pairs):
    item_tokens = []
    qty = None  # mặc định

    for token, tag in token_slot_pairs:
        if tag in ("B-item", "I-item"):
            item_tokens.append(token)
        elif tag == "B-quantity":
            try:
                if token.isdigit():
                    qty = int(token)
                else:
                    qty = w2n.word_to_num(token)
            except:
                continue

    item_name = " ".join(item_tokens) if item_tokens else None
    return item_name, qty or 1

def save_order(item, qty):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orders (item, qty) VALUES (?, ?)", (item, qty))
    conn.commit()
    conn.close()

def is_item_in_menu(item_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS menu (item_name TEXT UNIQUE)")
    cursor.execute("SELECT item_name FROM menu")
    menu = [r[0].lower() for r in cursor.fetchall()]
    conn.close()
    return item_name.lower() in menu

def handle_order_message(message):
    token_slot_pairs = predict_slots(message)
    item, qty = extract_entities(token_slot_pairs)

    if item:
        
        save_order(item, qty)
        return f"Order completed! Your order: {qty} {item}"
    else:
        return "Please try again"

# # ----- Ví dụ chạy thử -----
# if __name__ == "__main__":
#     while True:
#         msg = input("💬 Nhập yêu cầu đặt món: ")
#         if msg.lower() in ("exit", "quit"):
#             break
#         reply = handle_order_message(msg)
#         print("🤖", reply)
