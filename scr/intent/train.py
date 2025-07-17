from data_loader import ChatbotIntentDataLoader
from model_builder import ChatbotIntentModel
from trainer import Trainer
import pickle
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
os.makedirs('model', exist_ok=True)
loader = ChatbotIntentDataLoader(DATA_PATH)
X, y = loader.load_data()
train_ds, val_ds, test_ds, vectorizer, label_encoder = loader.preprocess(X, y)

num_classes = len(label_encoder.classes_)
builder = ChatbotIntentModel(max_features=10000, num_classes=num_classes)
model = builder.build_model()
trainer = Trainer(model, train_ds, val_ds, project_name="chatbot-demo")
trainer.train(epochs=50)
trainer.save_model("model/final_model.keras")

# Lưu vectorizer
vectorizer_config = vectorizer.get_config()
vectorizer_vocab = vectorizer.get_vocabulary()

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump((vectorizer_config, vectorizer_vocab), f)

# Lưu label encoder
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("Training complete")
