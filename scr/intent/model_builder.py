from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

class ChatbotIntentModel:
    def __init__(self, max_features=10000, num_classes=12, embedding_dim=64):
        self.max_features = max_features
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def build_model(self):
        model = Sequential([
            Embedding(input_dim=self.max_features + 1, output_dim=self.embedding_dim),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam',
            metrics=['accuracy']
        )

        return model
