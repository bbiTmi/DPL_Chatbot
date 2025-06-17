import pickle
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

# Load mô hình
model = load_model("model/final_model.h5")

# Load vectorizer
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer_config, vectorizer_vocab = pickle.load(f)

vectorizer = TextVectorization.from_config(vectorizer_config)
vectorizer.set_vocabulary(vectorizer_vocab)

# Load label encoder
label_encoder = joblib.load("model/label_encoder.pkl")

# Hàm dự đoán intent
def predict_intent(text: str) -> str:
    x = tf.constant([text])
    x_vector = vectorizer(x)
    prediction = model.predict(x_vector)
    pred_id = tf.argmax(prediction[0]).numpy()
    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent

# Chạy thử
if __name__ == "__main__":
    while True:
        user_input = input("Nhập câu hỏi (hoặc 'exit'): ")
        if user_input.lower() == "exit":
            break
        intent = predict_intent(user_input)
        print(f"→ Dự đoán intent: {intent}")
