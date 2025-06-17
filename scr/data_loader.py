import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class ChatbotIntentDataLoader:
    def __init__(self, data_path, max_features=10000, sequence_length=100, batch_size=32):
        self.data_path = data_path
        self.max_features = max_features
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vectorizer = TextVectorization(
            max_tokens=self.max_features,
            output_sequence_length=self.sequence_length,
            output_mode='int'
        )
        self.label_encoder = LabelEncoder()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        X = df['question']
        y = self.label_encoder.fit_transform(df['intent'])  # Encode text intent to integers
        return X, y

    def preprocess(self, X, y):
    # 1. Học từ vựng từ toàn bộ dữ liệu
        self.vectorizer.adapt(X.values)

        # 2. Chia dữ liệu thủ công (70/20/10)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=42)
        # (0.2222 của 90% sẽ là 20% tổng)

        # 3. Hàm helper tạo tf.data.Dataset
        def to_dataset(X_part, y_part):
            X_vec = self.vectorizer(X_part.values)
            return tf.data.Dataset.from_tensor_slices((X_vec, y_part)) \
                                .shuffle(1000) \
                                .batch(self.batch_size) \
                                .cache() \
                                .prefetch(tf.data.AUTOTUNE)

        # 4. Trả về 3 tập
        return (
            to_dataset(X_train, y_train),
            to_dataset(X_val, y_val),
            to_dataset(X_test, y_test),
            self.vectorizer,
            self.label_encoder
        )
