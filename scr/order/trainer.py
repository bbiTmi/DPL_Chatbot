from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class SlotFillingTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, project_name='slot-filling-bilstm'):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.project_name = project_name


    def train(self, epochs=50, batch_size=32):
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('slot_best_model.keras', save_best_only=True)
        ]

        history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks
        )
        return history

    def save_model(self, path='slot_model.keras'):
        self.model.save(path)
