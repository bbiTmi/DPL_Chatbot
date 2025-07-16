from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, TimeDistributed, Dense, LayerNormalization, Attention
from tensorflow.keras.layers import GlobalAveragePooling1D, concatenate
from tensorflow.keras.optimizers import Adam

class SlotFillingModel:
    def __init__(self, vocab_size, tag_size, embedding_dim=128, lstm_units=128, max_len=30):
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_len = max_len


    def build_model(self):
        inputs = Input(shape=(self.max_len,))

        # Embedding layer
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len, mask_zero=True)(inputs)
        x = Dropout(0.2)(x)

        # BiLSTM layers
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

        x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

        # TimeDistributed output
        outputs = TimeDistributed(Dense(self.tag_size, activation="softmax"))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

        return model
