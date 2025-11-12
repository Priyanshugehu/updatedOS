# ================================================================
# OS ALGORITHM PREDICTION PROJECT
# Predict Best Page Replacement or Scheduling Algorithm
# ================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from IPython.display import display, clear_output
import ipywidgets as widgets

# ================================================================
# PART 1: PAGE REPLACEMENT ALGORITHM PREDICTION
# ================================================================
print("🔹 Generating Page Replacement Dataset...")

np.random.seed(42)
data = []
samples = 2000

for _ in range(samples):
    num_pages = np.random.randint(5, 50)
    memory_size = np.random.randint(2, 12)
    locality = np.round(np.random.uniform(0.05, 0.95), 2)
    sequence_len = np.random.randint(20, 200)

    frames_to_pages = memory_size / num_pages
    freq_var = np.random.uniform(0.1, 1.0) * (1 - locality)

    # rule-based label
    if locality < 0.3 and frames_to_pages < 0.25:
        best_algo = "FIFO"
    elif 0.3 <= locality < 0.7:
        best_algo = "LRU"
    else:
        best_algo = "Optimal"

    data.append([num_pages, memory_size, locality, sequence_len,
                 frames_to_pages, freq_var, best_algo])

df_page = pd.DataFrame(data, columns=[
    "num_pages", "memory_size", "locality", "sequence_len",
    "frames_to_pages", "freq_var", "best_algo"
])

print("Page Replacement Dataset Created.")
print(df_page["best_algo"].value_counts())

# ================================================================
# PREPROCESS PAGE DATA
# ================================================================
features_page = ["num_pages", "memory_size", "locality", "sequence_len",
                 "frames_to_pages", "freq_var"]
X_page = df_page[features_page]
y_page = df_page["best_algo"]

le_page = LabelEncoder()
y_page_encoded = le_page.fit_transform(y_page)

scaler_page = StandardScaler()
X_page_scaled = scaler_page.fit_transform(X_page)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_page_scaled, y_page_encoded, test_size=0.2, random_state=42, stratify=y_page_encoded
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
}

results_page = {}
for name, model in models.items():
    model.fit(X_train_p, y_train_p)
    y_pred = model.predict(X_test_p)
    acc = accuracy_score(y_test_p, y_pred)
    results_page[name] = (model, acc)
    print(f"\n{name} Accuracy (Page): {acc:.3f}")
    print(classification_report(y_test_p, y_pred, target_names=le_page.classes_))

best_page_name = max(results_page, key=lambda k: results_page[k][1])
best_page_model = results_page[best_page_name][0]
print(f"✅ Best Page Replacement Model: {best_page_name}")

joblib.dump(best_page_model, "best_page_model.pkl")
joblib.dump(le_page, "label_encoder_page.pkl")
joblib.dump(scaler_page, "scaler_page.pkl")

# ================================================================
# PART 2: CPU SCHEDULING ALGORITHM PREDICTION
# ================================================================
print("\n🔹 Generating Scheduling Algorithm Dataset...")

data_sched = []
samples = 2000

for _ in range(samples):
    num_processes = np.random.randint(3, 20)
    avg_burst = np.random.randint(2, 20)
    avg_arrival_gap = np.random.uniform(0.5, 5.0)
    priority_variance = np.random.uniform(0.1, 1.0)
    quantum = np.random.randint(2, 10)

    cpu_util = np.random.uniform(0.4, 0.95)
    io_bound_ratio = np.random.uniform(0.1, 0.9)

    # Derived rules for labeling
    if io_bound_ratio > 0.6 and quantum > 5:
        best_algo = "Round Robin"
    elif avg_burst < 6 and avg_arrival_gap > 2.5:
        best_algo = "SJF"
    else:
        best_algo = "FCFS"

    data_sched.append([num_processes, avg_burst, avg_arrival_gap,
                       priority_variance, quantum, cpu_util,
                       io_bound_ratio, best_algo])

df_sched = pd.DataFrame(data_sched, columns=[
    "num_processes", "avg_burst", "avg_arrival_gap", "priority_variance",
    "quantum", "cpu_util", "io_bound_ratio", "best_algo"
])

print("Scheduling Dataset Created.")
print(df_sched["best_algo"].value_counts())

# ================================================================
# PREPROCESS SCHEDULING DATA
# ================================================================
features_sched = ["num_processes", "avg_burst", "avg_arrival_gap",
                  "priority_variance", "quantum", "cpu_util", "io_bound_ratio"]
X_sched = df_sched[features_sched]
y_sched = df_sched["best_algo"]

le_sched = LabelEncoder()
y_sched_encoded = le_sched.fit_transform(y_sched)

scaler_sched = StandardScaler()
X_sched_scaled = scaler_sched.fit_transform(X_sched)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sched_scaled, y_sched_encoded, test_size=0.2, random_state=42, stratify=y_sched_encoded
)

results_sched = {}
for name, model in models.items():
    model.fit(X_train_s, y_train_s)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test_s, y_pred)
    results_sched[name] = (model, acc)
    print(f"\n{name} Accuracy (Scheduling): {acc:.3f}")
    print(classification_report(y_test_s, y_pred, target_names=le_sched.classes_))

best_sched_name = max(results_sched, key=lambda k: results_sched[k][1])
best_sched_model = results_sched[best_sched_name][0]
print(f"✅ Best Scheduling Model: {best_sched_name}")

joblib.dump(best_sched_model, "best_sched_model.pkl")
joblib.dump(le_sched, "label_encoder_sched.pkl")
joblib.dump(scaler_sched, "scaler_sched.pkl")

# ================================================================
# INTERACTIVE PREDICTION SECTION
# ================================================================
print("\n✅ Models trained and saved successfully!")

# Load models
page_model = joblib.load("best_page_model.pkl")
sched_model = joblib.load("best_sched_model.pkl")
le_page = joblib.load("label_encoder_page.pkl")
le_sched = joblib.load("label_encoder_sched.pkl")
scaler_page = joblib.load("scaler_page.pkl")
scaler_sched = joblib.load("scaler_sched.pkl")

# ---------------- PAGE PREDICTION UI ----------------
print("\n📘 Interactive Page Replacement Predictor")
num_pages = widgets.IntSlider(value=10, min=5, max=50, description="Pages:")
memory_size = widgets.IntSlider(value=4, min=2, max=12, description="Frames:")
locality = widgets.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description="Locality:")
sequence_len = widgets.IntSlider(value=100, min=20, max=200, step=10, description="Seq Len:")
out_page = widgets.Output()

def predict_page(btn):
    with out_page:
        clear_output()
        frames_to_pages = memory_size.value / num_pages.value
        freq_var = 0.8 * (1 - locality.value)
        df_input = pd.DataFrame([[num_pages.value, memory_size.value, locality.value, sequence_len.value,
                                  frames_to_pages, freq_var]], columns=features_page)
        scaled = scaler_page.transform(df_input)
        pred = page_model.predict(scaled)
        print("Predicted Page Replacement Algorithm:", le_page.inverse_transform(pred)[0])

btn_page = widgets.Button(description="Predict Page Algo", button_style="success")
btn_page.on_click(predict_page)
display(num_pages, memory_size, locality, sequence_len, btn_page, out_page)

# ---------------- SCHEDULING PREDICTION UI ----------------
print("\n⚙️ Interactive Scheduling Algorithm Predictor")
num_processes = widgets.IntSlider(value=5, min=3, max=20, description="Processes:")
avg_burst = widgets.IntSlider(value=5, min=1, max=20, description="Avg Burst:")
avg_arrival_gap = widgets.FloatSlider(value=2.0, min=0.5, max=5.0, step=0.5, description="Arrival Gap:")
priority_variance = widgets.FloatSlider(value=0.5, min=0.1, max=1.0, step=0.1, description="Priority Var:")
quantum = widgets.IntSlider(value=4, min=2, max=10, description="Quantum:")
cpu_util = widgets.FloatSlider(value=0.75, min=0.4, max=0.95, step=0.05, description="CPU Util:")
io_bound_ratio = widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.1, description="I/O Ratio:")
out_sched = widgets.Output()

def predict_sched(btn):
    with out_sched:
        clear_output()
        df_input = pd.DataFrame([[num_processes.value, avg_burst.value, avg_arrival_gap.value,
                                  priority_variance.value, quantum.value, cpu_util.value,
                                  io_bound_ratio.value]], columns=features_sched)
        scaled = scaler_sched.transform(df_input)
        pred = sched_model.predict(scaled)
        print("Predicted Scheduling Algorithm:", le_sched.inverse_transform(pred)[0])

btn_sched = widgets.Button(description="Predict Scheduling Algo", button_style="info")
btn_sched.on_click(predict_sched)
display(num_processes, avg_burst, avg_arrival_gap, priority_variance,
        quantum, cpu_util, io_bound_ratio, btn_sched, out_sched)
