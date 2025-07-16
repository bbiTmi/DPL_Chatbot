from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, TimeDistributed, Dense

class SlotFillingModel:
    def __init__(self, vocab_size, tag_size, embedding_dim=64, lstm_units=64, max_len=30):
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_len = max_len

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, 
                            input_length=self.max_len, mask_zero=True))
        model.add(Bidirectional(LSTM(units=self.lstm_units, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(TimeDistributed(Dense(self.tag_size, activation="softmax")))

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model
