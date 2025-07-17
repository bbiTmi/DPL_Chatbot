import pickle
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
from keras.layers import TFSMLayer
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "test_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_formatted")
VECTOR_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")


with open(VECTOR_PATH, "rb") as f:
    vectorizer_config, vectorizer_vocab = pickle.load(f)

vectorizer = TextVectorization.from_config(vectorizer_config)
vectorizer.set_vocabulary(vectorizer_vocab)
vectorizer.output_mode = "int"


label_encoder = joblib.load(LABEL_ENCODER_PATH)


df_test = pd.read_csv(DATA_PATH)
questions = df_test["question"].tolist()
true_labels = df_test["intent"].tolist()

# ✅ Encode labels → chỉ số
true_label_ids = label_encoder.transform(true_labels)

# ✅ Vector hóa đầu vào
X_test = vectorizer(tf.constant(questions))
X_test = tf.cast(X_test, dtype=tf.float32)  # 🧠 Rất quan trọng!

# ✅ Dự đoán với TFSMLayer → trả về dict
raw_output = model(X_test)
logits = raw_output["dense_1"].numpy()

# ✅ Lấy nhãn dự đoán
pred_ids = np.argmax(logits, axis=1)
pred_labels = label_encoder.inverse_transform(pred_ids)

# ✅ In báo cáo đánh giá
print(classification_report(true_labels, pred_labels, digits=4))
