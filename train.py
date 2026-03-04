import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

print("Loading dataset...")

data = load_wine()
X = data.data
y = data.target

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "model.pkl")

metrics = {"accuracy": accuracy}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Model and metrics saved.")
