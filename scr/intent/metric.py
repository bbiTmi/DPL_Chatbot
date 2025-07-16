import pickle
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

model = load_model("model/final_model.h5")


with open("model/vectorizer.pkl", "rb") as f:
    vectorizer_config, vectorizer_vocab = pickle.load(f)

vectorizer = TextVectorization.from_config(vectorizer_config)
vectorizer.set_vocabulary(vectorizer_vocab)
vectorizer.output_mode = "int"


label_encoder = joblib.load("model/label_encoder.pkl")


df_test = pd.read_csv("data/test_data.csv")
questions = df_test["question"].tolist()
true_labels = df_test["intent"].tolist()


X_test = vectorizer(tf.constant(questions))


pred_probs = model.predict(X_test)
pred_indices = np.argmax(pred_probs, axis=1)
pred_labels = label_encoder.inverse_transform(pred_indices)

print(classification_report(true_labels, pred_labels, digits=4))
