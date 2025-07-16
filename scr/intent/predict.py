import pickle
import joblib
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.keras")
VECTOR_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

model = load_model(MODEL_PATH)

# Load vectorize
with open(VECTOR_PATH, "rb") as f:
    vectorizer_config, vectorizer_vocab = pickle.load(f)

vectorizer = TextVectorization.from_config(vectorizer_config)
vectorizer.set_vocabulary(vectorizer_vocab)
vectorizer.output_mode = "int"  # ✅ để trả về chỉ số integer cho Embedding

label_encoder = joblib.load(LABEL_ENCODER_PATH)

def predict_intent(message, threshold=0.6):
    x_input = vectorizer(tf.constant([message]))
    pred = model.predict(x_input)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])

    if confidence < threshold:
        return "unknown", confidence  # hoặc "fallback"

    intent = label_encoder.inverse_transform([idx])[0]
    return intent, confidence

# if __name__ == "__main__":
#     while True:
#         user_input = input("Enter question ('exit' to break): ")
#         if user_input.lower() == "exit":
#             break
#         intent, conf = predict_intent(user_input)
#         print(f"→ Intent: {intent} (confidence = {conf:.2f})")
