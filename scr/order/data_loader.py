import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import joblib

class SlotFillingDataLoader:
    def __init__(self, data_path, max_len=30):
        self.data_path = data_path
        self.max_len = max_len
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = {}

    def load_data(self):
        df = pd.read_csv(self.data_path)

        # Convert string to list
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
        df["slots"] = df["slots"].apply(ast.literal_eval)

        # Vocabulary for words and tags
        all_words = sorted({w for tokens in df["tokens"] for w in tokens})
        all_tags = sorted({t for tags in df["slots"] for t in tags})

        self.word2idx = {w: i + 2 for i, w in enumerate(all_words)}
        self.word2idx["PAD"] = 0
        self.word2idx["UNK"] = 1

        self.tag2idx = {t: i + 1 for i, t in enumerate(all_tags)}
        self.tag2idx["PAD"] = 0

        self.idx2tag = {i: t for t, i in self.tag2idx.items()}

        # Encode
        X = [[self.word2idx.get(w, self.word2idx["UNK"]) for w in s] for s in df["tokens"]]
        y = [[self.tag2idx[t] for t in s] for s in df["slots"]]

        # Padding
        X = pad_sequences(X, maxlen=self.max_len, padding='post')
        y = pad_sequences(y, maxlen=self.max_len, padding='post')

        y = np.array([to_categorical(i, num_classes=len(self.tag2idx)) for i in y])
        return X, y

    def split_data(self, X, y, val_size=0.1, test_size=0.1):
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_metadata(self, word2idx_path, tag2idx_path):
        joblib.dump(self.word2idx, word2idx_path)
        joblib.dump(self.idx2tag, tag2idx_path)
