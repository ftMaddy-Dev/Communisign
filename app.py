from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import joblib
import os
from collections import deque, Counter
from features import extract_normalized_landmarks_from_dicts, engineer_features, COLUMNS

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
for path in [".venv/sign_model.pkl", "sign_model.pkl"]:
    if os.path.exists(path):
        model = joblib.load(path)
        le    = joblib.load(path.replace("sign_model", "label_encoder"))
        print(f"✅ Model loaded: {path}  ({model.n_features_in_} features)")
        MODEL_LOADED = True
        break
else:
    print("⚠️  sign_model.pkl not found — retrain first")
    MODEL_LOADED = False

CONFIDENCE_THRESHOLD = 0.80
SMOOTH_WINDOW        = 10
MIN_VOTES            = 3           # ← lowered from 5 for snappier response

prediction_buffer = deque(maxlen=SMOOTH_WINDOW)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    global prediction_buffer
    if not MODEL_LOADED:
        return jsonify({"letter": "No model", "confidence": 0})

    data    = request.get_json(force=True)
    raw_lms = data.get("landmarks", [])

    if len(raw_lms) != 21:
        return jsonify({"letter": "-", "confidence": 0})

    flat63 = extract_normalized_landmarks_from_dicts(raw_lms)
    if flat63 is None:
        return jsonify({"letter": "-", "confidence": 0})

    feat84   = engineer_features(flat63)
    features = pd.DataFrame([feat84], columns=COLUMNS)

    prediction = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    confidence = float(max(proba))
    label      = le.inverse_transform([prediction])[0]

    if confidence >= CONFIDENCE_THRESHOLD:
        prediction_buffer.append(label)

    if len(prediction_buffer) >= MIN_VOTES:
        smoothed = Counter(prediction_buffer).most_common(1)[0][0]
        return jsonify({"letter": smoothed, "confidence": round(confidence * 100, 1)})

    return jsonify({"letter": "Warming up...", "confidence": round(confidence * 100, 1)})


@app.route("/clear_buffer", methods=["POST"])
def clear_buffer():
    """Called by JS when no hand is visible — prevents stale votes."""
    prediction_buffer.clear()
    return jsonify({"ok": True})


@app.route("/predict")
def predict():
    return jsonify({"letter": "A", "confidence": 95})


if __name__ == "__main__":
    app.run(debug=True)
    @app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200
