import os
import pickle
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# Đường dẫn
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.keras")
VECTOR_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

# Load mô hình SavedModel bằng TFSMLayer
model = tf.keras.models.load_model(MODEL_PATH)

# Load vectorizer
with open(VECTOR_PATH, "rb") as f:
    vectorizer_config, vectorizer_vocab = pickle.load(f)

vectorizer = TextVectorization.from_config(vectorizer_config)
vectorizer.set_vocabulary(vectorizer_vocab)
vectorizer.output_mode = "int"

# Load label encoder
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Hàm dự đoán intent
def predict_intent(message, threshold=0.6):
    x_input = vectorizer(tf.constant([message]))  # shape (1, max_len)
    x_input = tf.cast(x_input, tf.int32)

    pred = model(x_input).numpy()
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])

    if confidence < threshold:
        return "unknown", confidence

    intent = label_encoder.inverse_transform([idx])[0]
    return intent, confidence

# # CLI thử nghiệm
# if __name__ == "__main__":
#     while True:
#         user_input = input("❓ Nhập câu hỏi ('exit' để thoát): ")
#         if user_input.lower() == "exit":
#             break
#         intent, conf = predict_intent(user_input)
#         print(f"→ 🎯 Intent: {intent} (conf = {conf:.2f})")
