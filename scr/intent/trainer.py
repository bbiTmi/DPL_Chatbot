import wandb
from wandb.integration.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class Trainer:
    def __init__(self, model, train_data, val_data, project_name='chatbot-demo'):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.project_name = project_name

        # Init wandb
        # wandb.init(project=self.project_name, config={
        #     "architecture": model.name if hasattr(model, 'name') else "LSTM",
        #     "epochs": 7,
        #     "batch_size": train_data._batch_size if hasattr(train_data, '_batch_size') else 32,
        # })

    def train(self, epochs=7):
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('best_model.keras', save_best_only=True)
            #WandbCallback(log_weights=True, log_graph=False)
        ]
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    def save_model(self, path='intent_model.keras'):
        # Lưu mô hình vào file
        self.model.save(path)

        # Dùng artifact để log model mà không cần tạo symlink (fix lỗi trên Windows)
        # artifact = wandb.Artifact(name='intent-model', type='model')
        # artifact.add_file(path)
        # wandb.log_artifact(artifact)

