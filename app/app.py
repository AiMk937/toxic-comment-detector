"""Flask web app and JSON API for the toxic comment detector.

Run with:  python -m app.app   (then open http://127.0.0.1:5000)
"""
import os

from flask import Flask, jsonify, render_template, request

from src.config import LABEL_NAMES, MODEL_PATH
from src.predict import ToxicityDetector

MAX_COMMENTS = 50
MAX_CHARS = 5000

app = Flask(__name__, static_folder="static", template_folder="templates")
app.json.sort_keys = False  # Keeps labels in toxic -> identity_hate order

# Loads the model once at startup; the app still starts (with a clear error) if training hasn't been run
try:
    detector = ToxicityDetector(os.getenv("MODEL_PATH", MODEL_PATH))
    load_error = None
except FileNotFoundError as err:
    detector, load_error = None, str(err)


@app.get("/")
def home():
    return render_template("index.html", label_names=LABEL_NAMES)


@app.get("/health")
def health():
    return jsonify({"status": "ok" if detector else "model_missing", "error": load_error})


@app.post("/predict")
def predict():
    if detector is None:
        return jsonify({"error": load_error}), 503

    # Validates input size so one request can't overload the server
    data = request.get_json(silent=True) or {}
    comments = data.get("comments")
    if not isinstance(comments, list) or not comments:
        return jsonify({"error": "Send JSON like {\"comments\": [\"text\", ...]}"}), 400
    comments = [str(c)[:MAX_CHARS] for c in comments if str(c).strip()][:MAX_COMMENTS]
    if not comments:
        return jsonify({"error": "All comments were empty."}), 400

    threshold = float(data.get("threshold", 0.5))
    detector.threshold = min(max(threshold, 0.05), 0.95)
    return jsonify({"predictions": detector.predict(comments)})


if __name__ == "__main__":
    # Debug mode only when explicitly enabled - it allows code execution if exposed
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
