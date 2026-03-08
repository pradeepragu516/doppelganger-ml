
# ==============================
# AI-Powered Predictive Maintenance
# Anomaly Detection Training Script
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report

import joblib


# -------------------------------
# 1. Load Dataset
# -------------------------------

print("Loading dataset...")
import os

# Look for dataset in multiple locations (dataset/ subfolder preferred)
possible_paths = [
    os.path.join('dataset', 'ai4i2020.csv'),
    os.path.join('dataset', 'sensor_dataset.csv'),
    'ai4i2020.csv',
    'sensor_dataset.csv'
]

data_path = None
for p in possible_paths:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError(f"No dataset found. Checked: {possible_paths}")

print(f"Loading dataset from: {data_path}")
data = pd.read_csv(data_path)

print("\nDataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())


# -------------------------------
# 2. Data Cleaning
# -------------------------------

print("\nCleaning dataset...")

# Drop unnecessary columns
data = data.drop(['UDI','Product ID'], axis=1)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# -------------------------------
# 3. Encode Categorical Features
# -------------------------------

print("\nEncoding categorical features...")

encoder = LabelEncoder()

data['Type'] = encoder.fit_transform(data['Type'])


# -------------------------------
# 4. Feature Engineering
# -------------------------------

# Temperature difference feature
data['temp_difference'] = data['Process temperature [K]'] - data['Air temperature [K]']


# -------------------------------
# 5. Define Features & Target
# -------------------------------

X = data.drop('Machine failure', axis=1)
y = data['Machine failure']


# -------------------------------
# 6. Normalize Data
# -------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -------------------------------
# 7. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# 8. Train Anomaly Detection Model
# -------------------------------

print("\nTraining Isolation Forest model...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_train)


# -------------------------------
# 9. Predict Anomalies
# -------------------------------

print("\nPredicting anomalies...")

predictions = model.predict(X_test)

# Convert output format
predictions = np.where(predictions == -1, 1, 0)


# -------------------------------
# 10. Evaluate Model
# -------------------------------

print("\nModel Evaluation:")

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -------------------------------
# 11. Detect Anomalies in Dataset
# -------------------------------

data['anomaly'] = model.predict(X_scaled)

anomalies = data[data['anomaly'] == -1]

print("\nTotal anomalies detected:", len(anomalies))


# -------------------------------
# 12. Visualization
# -------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    data['Rotational speed [rpm]'],
    data['Torque [Nm]'],
    c=data['anomaly']
)

plt.title("Machine Sensor Anomaly Detection")
plt.xlabel("Rotational Speed")
plt.ylabel("Torque")

plt.show()


# -------------------------------
# 13. Save Model
# -------------------------------

print("\nSaving trained model...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, os.path.join('models', 'anomaly_model.pkl'))
joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))

print("Model saved successfully in ./models/")


# -------------------------------
# 14. Test Sample Prediction
# -------------------------------

print("\nTesting sample prediction...")

# Sample with all 12 features:
# Type (encoded: L=0, M=1, H=2), Air temp, Process temp, 
# Rotational speed, Torque, Tool wear, 
# TWF, HDF, PWF, OSF, RNF, temp_difference
sample = [[1, 298.1, 308.6, 1551, 70, 200, 0, 0, 0, 0, 0, 10.5]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

if prediction[0] == -1:
    print("⚠ Anomaly Detected! Maintenance Required")
else:
    print("✅ Machine Operating Normally")

