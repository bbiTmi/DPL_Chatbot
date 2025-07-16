from data_loader import SlotFillingDataLoader
from model_builder import SlotFillingModel
from trainer import SlotFillingTrainer
import os
import joblib

os.makedirs('model', exist_ok=True)

loader = SlotFillingDataLoader(data_path="data/data-slot-filling.csv", max_len=30)
X, y = loader.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)

vocab_size = len(loader.word2idx)
tag_size = len(loader.tag2idx)
builder = SlotFillingModel(vocab_size=vocab_size, tag_size=tag_size, max_len=30)
model = builder.build_model()

trainer = SlotFillingTrainer(model, X_train, y_train, X_val, y_val, project_name="slot-filling-bilstm")
trainer.train(epochs=30)
trainer.save_model("model/slot_model.h5")

joblib.dump(loader.word2idx, "model/word2idx.pkl")
joblib.dump(loader.idx2tag, "model/idx2tag.pkl")

