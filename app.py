from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# LOAD MODEL ARTIFACTS
model = joblib.load("artifacts/xgb_model_raw.pkl")
scaler = joblib.load("artifacts/scaler_raw.pkl")
encoder = joblib.load("artifacts/encoder_raw.pkl")
feature_cols = joblib.load("artifacts/feature_cols_raw.pkl")

print("✅ TypeID API loaded – Ready")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "TypeID API running"})

@app.route("/predict_sequential", methods=["POST"])
def predict_sequential():
    try:
        data = request.json
        features = []

        for col in feature_cols:
            val = float(data.get(col, 0))

            if col == "ks_count":
                val = max(val, 150)
            elif col == "ks_rate":
                val = min(max(val, 1), 200)
            elif col in ["dwell_mean", "flight_mean", "digraph_mean"]:
                val = min(max(val, 10), 2000)
            elif col in ["dwell_std", "flight_std", "digraph_std"]:
                val = min(max(val, 1), 1000)
            elif col == "backspace_rate":
                val = min(max(val, 0), 1)
            elif col == "wps":
                val = min(max(val, 0.1), 50)
            elif col == "wpm":
                val = min(max(val, 1), 200)

            features.append(val)

        features_scaled = scaler.transform([features])
        probabilities = model.predict_proba(features_scaled)[0]

        top3_idx = np.argsort(probabilities)[-3:][::-1]

        top3 = []
        for i, idx in enumerate(top3_idx):
            top3.append({
                "rank": i + 1,
                "user": encoder.classes_[idx],
                "confidence": float(probabilities[idx] * 100)
            })

        return jsonify({
            "success": True,
            "top3_predictions": top3,
            "best_user": top3[0]["user"],
            "confidence": top3[0]["confidence"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
