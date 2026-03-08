from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Attempt to load model and scaler from models/
MODEL_PATH = os.path.join('models', 'anomaly_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not available'}), 503

    data = request.json or {}
    temperature = float(data.get('temperature', 65))
    pressure = float(data.get('pressure', 90))
    vibration = float(data.get('vibration', 3))

    # Build input; if scaler expects 3 features, pass as-is. Otherwise, build vector with zeros.
    expected = getattr(scaler, 'n_features_in_', None)
    if expected == 3:
        features = np.array([[temperature, pressure, vibration]])
    else:
        # try to map using dataset columns if present
        dataset_path = None
        for p in [os.path.join('dataset', 'ai4i2020.csv'), os.path.join('dataset', 'sensor_dataset.csv')]:
            if os.path.exists(p):
                dataset_path = p
                break

        if dataset_path:
            df = pd.read_csv(dataset_path)
            cols = list(getattr(scaler, 'feature_names_in_', df.columns[:expected]))
            row = np.zeros(len(cols))
            # heuristic mapping
            for i, col in enumerate(cols):
                cl = col.lower()
                if 'temp' in cl and row[i] == 0:
                    row[i] = temperature
                elif 'torque' in cl or 'press' in cl:
                    row[i] = pressure
                elif 'rotat' in cl or 'rpm' in cl or 'vib' in cl:
                    row[i] = vibration
            features = row.reshape(1, -1)
        else:
            # fallback: pad zeros
            features = np.zeros((1, expected))
            features[0, 0:3] = [temperature, pressure, vibration]

    # scale and predict
    try:
        X_scaled = scaler.transform(features)
        pred = model.predict(X_scaled)
        # decision_function: higher = more normal, lower = more anomalous
        try:
            score = float(model.decision_function(X_scaled)[0])
        except Exception:
            score = None

        # Interpret prediction: IsolationForest typically returns -1 for anomalies
        status = 'Anomaly' if int(pred[0]) == -1 else 'Normal'

        response = {'status': status}
        if score is not None:
            response['score'] = score

        # Echo metadata if provided
        meta = {}
        for k in ['machineId', 'location', 'temperature', 'pressure', 'vibration']:
            if k in data:
                meta[k] = data.get(k)
        if meta:
            response['received'] = meta

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)