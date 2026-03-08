import joblib
import numpy as np
import os
import pandas as pd

# Load trained model and scaler
model = joblib.load("models/anomaly_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def _find_dataset():
    candidates = [
        os.path.join('dataset', 'ai4i2020.csv'),
        os.path.join('dataset', 'sensor_dataset.csv'),
        'ai4i2020.csv',
        'sensor_dataset.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _map_input_to_features(temperature, pressure, vibration, feature_names):
    # Create a prototype row using dataset means if available, otherwise zeros
    dataset_path = _find_dataset()
    if dataset_path:
        df = pd.read_csv(dataset_path)
        numeric_means = df.select_dtypes(include=[float, int]).mean()
    else:
        numeric_means = pd.Series(dtype=float)

    # initialize values dict with means or zeros
    vals = {fn: numeric_means.get(fn, 0.0) for fn in feature_names}

    # Heuristic mapping
    fn_lower = [fn.lower() for fn in feature_names]

    # temperature -> prefer 'air temperature', then 'process temperature', then any 'temp'
    temp_idx = None
    for key in ['air temperature', 'process temperature', 'temperature', 'temp']:
        for i, fn in enumerate(fn_lower):
            if key in fn:
                temp_idx = feature_names[i]
                break
        if temp_idx:
            break

    # vibration -> prefer 'rotational' or 'rpm' or 'vibration'
    vib_idx = None
    for key in ['rotational', 'rpm', 'vibration']:
        for i, fn in enumerate(fn_lower):
            if key in fn:
                vib_idx = feature_names[i]
                break
        if vib_idx:
            break

    # pressure -> prefer 'torque' or 'pressure'
    press_idx = None
    for key in ['torque', 'pressure']:
        for i, fn in enumerate(fn_lower):
            if key in fn:
                press_idx = feature_names[i]
                break
        if press_idx:
            break

    # Fallback assignments: if None, fill first available numeric columns
    if temp_idx is None:
        for fn in feature_names:
            if fn in numeric_means.index or True:
                temp_idx = feature_names[0]
                break
    if vib_idx is None:
        vib_idx = feature_names[1] if len(feature_names) > 1 else feature_names[0]
    if press_idx is None:
        press_idx = feature_names[2] if len(feature_names) > 2 else feature_names[0]

    vals[temp_idx] = temperature
    vals[press_idx] = pressure
    vals[vib_idx] = vibration

    # Return a 2D array in the order of feature_names
    row = [float(vals[fn]) for fn in feature_names]
    return np.array([row])


def predict_anomaly(temperature, pressure, vibration):
    # Determine expected feature count/names from scaler
    expected = getattr(scaler, 'n_features_in_', None)
    feature_names = None
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)

    if feature_names is None and expected is not None:
        # create placeholder names
        feature_names = [f'feature_{i}' for i in range(expected)]

    # If scaler expects 3 features, use simple path
    if expected == 3:
        input_data = np.array([[temperature, pressure, vibration]])
    else:
        input_data = _map_input_to_features(temperature, pressure, vibration, feature_names)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # IsolationForest output: -1 = anomaly, 1 = normal
    return "Anomaly" if prediction[0] == -1 else "Normal"


if __name__ == "__main__":
    result = predict_anomaly(temperature=80, pressure=40, vibration=10)
    print("Prediction:", result)