import wandb
from wandb.integration.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class SlotFillingTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, project_name='slot-filling-bilstm'):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.project_name = project_name

        # Khởi tạo Weights & Biases
        wandb.init(project=self.project_name, config={
            "architecture": model.name if hasattr(model, 'name') else "BiLSTM",
            "epochs": 50,
            "batch_size": 32,
            "embedding_dim": model.layers[0].output_dim if hasattr(model.layers[0], 'output_dim') else 64,
            "lstm_units": model.layers[1].units if hasattr(model.layers[1], 'units') else 64,
        })

    def train(self, epochs=50, batch_size=32):
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('slot_best_model.h5', save_best_only=True),
            WandbCallback(log_weights=True)
        ]

        history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks
        )
        return history

    def save_model(self, path='slot_model.h5'):
        self.model.save(path)

        artifact = wandb.Artifact(name='slot-model', type='model')
        artifact.add_file(path)
        wandb.log_artifact(artifact)
