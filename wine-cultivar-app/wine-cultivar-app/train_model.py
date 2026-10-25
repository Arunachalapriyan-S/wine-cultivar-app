"""Train the wine cultivar classifier and save it for the Streamlit app.

Run once:  python train_model.py
Creates:   model.joblib  (scikit-learn Pipeline)
           model_meta.json (feature order, slider ranges, class names, accuracy)
"""
import json

import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Feature ORDER matters: the app must build its input row in exactly this order.
FEATURES = ["alcohol", "malic_acid", "total_phenols", "flavanoids", "color_intensity", "proline"]

data = load_wine(as_frame=True)
X = data.frame[FEATURES]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, random_state=42),
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Test accuracy: {acc:.3f}")
print(classification_report(y_test, pred, target_names=data.target_names))

joblib.dump(model, "model.joblib")

meta = {
    "features": FEATURES,
    "classes": [f"Cultivar {i + 1}" for i in range(len(data.target_names))],
    "accuracy": round(float(acc), 3),
    "n_samples": int(len(X)),
    "ranges": {
        f: {
            "min": round(float(X[f].min()), 2),
            "max": round(float(X[f].max()), 2),
            "default": round(float(X[f].median()), 2),
        }
        for f in FEATURES
    },
}
with open("model_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print("Saved model.joblib and model_meta.json")
