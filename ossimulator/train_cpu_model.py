import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df_cpu = pd.read_csv('cpu_scheduling_dataset.csv')

# Features
features_cpu = [
    "num_processes", "avg_burst_time", "avg_arrival_time",
    "avg_waiting_time", "cpu_utilization", "time_quantum",
    "burst_to_quantum", "waiting_to_arrival"
]
X_cpu = df_cpu[features_cpu]
y_cpu = df_cpu["best_algo"]

le_cpu = LabelEncoder()
y_cpu_encoded = le_cpu.fit_transform(y_cpu)

scaler_cpu = StandardScaler()
X_cpu_scaled = scaler_cpu.fit_transform(X_cpu)

X_train_cpu, X_test_cpu, y_train_cpu, y_test_cpu = train_test_split(
    X_cpu_scaled, y_cpu_encoded, test_size=0.2, random_state=42, stratify=y_cpu_encoded
)

# Train models
models_cpu = {
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
}

results_cpu = {}
for name, model in models_cpu.items():
    model.fit(X_train_cpu, y_train_cpu)
    y_pred_cpu = model.predict(X_test_cpu)
    acc = accuracy_score(y_test_cpu, y_pred_cpu)
    results_cpu[name] = (model, acc)
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=le_cpu.classes_))

best_name_cpu = max(results_cpu, key=lambda k: results_cpu[k][1])
best_model_cpu = results_cpu[best_name_cpu][0]
print(f"\nSelected best model for CPU: {best_name_cpu} (acc={results_cpu[best_name_cpu][1]:.3f})")

# Save
joblib.dump(best_model_cpu, "best_cpu_scheduling_model.pkl")
joblib.dump(scaler_cpu, "scaler_cpu.pkl")
joblib.dump(le_cpu, "label_encoder_cpu.pkl")
print("Saved: best_cpu_scheduling_model.pkl, scaler_cpu.pkl, label_encoder_cpu.pkl")
