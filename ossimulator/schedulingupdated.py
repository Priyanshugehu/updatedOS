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
>>>>>>> 164492b38d85f20bd4b59d20d12867b6f15b82c7
=======
# ================================================================
# 🧠 OSync OS Algorithm Simulation Functions
# Contains implementations of Page Replacement and CPU Scheduling
# algorithms to run after ML prediction.
# ================================================================

import numpy as np

# ================================================================
# 🔹 PAGE REPLACEMENT ALGORITHMS
# ================================================================

def fifo_page_replacement(pages, frames):
    """Simulate FIFO Page Replacement"""
    memory = []
    page_faults = 0
    for page in pages:
        if page not in memory:
            if len(memory) < frames:
                memory.append(page)
            else:
                memory.pop(0)
                memory.append(page)
            page_faults += 1
    return page_faults


def lru_page_replacement(pages, frames):
    """Simulate LRU Page Replacement"""
    memory = []
    page_faults = 0
    recently_used = []
    for page in pages:
        if page not in memory:
            if len(memory) < frames:
                memory.append(page)
            else:
                lru_page = recently_used.pop(0)
                memory.remove(lru_page)
                memory.append(page)
            page_faults += 1
        if page in recently_used:
            recently_used.remove(page)
        recently_used.append(page)
    return page_faults


def optimal_page_replacement(pages, frames):
    """Simulate Optimal Page Replacement"""
    memory = []
    page_faults = 0
    for i in range(len(pages)):
        if pages[i] not in memory:
            if len(memory) < frames:
                memory.append(pages[i])
            else:
                future = pages[i + 1:]
                indices = []
                for x in memory:
                    if x in future:
                        indices.append(future.index(x))
                    else:
                        indices.append(float('inf'))
                memory.pop(indices.index(max(indices)))
                memory.append(pages[i])
            page_faults += 1
    return page_faults


# ================================================================
# ⚙️ CPU SCHEDULING ALGORITHMS
# ================================================================

def fcfs_scheduling(processes, burst_times):
    """Simulate First Come First Serve Scheduling"""
    waiting_times = [0]
    for i in range(1, len(burst_times)):
        waiting_times.append(waiting_times[-1] + burst_times[i - 1])
    avg_waiting = sum(waiting_times) / len(waiting_times)
    turnaround_times = [wt + bt for wt, bt in zip(waiting_times, burst_times)]
    avg_turnaround = sum(turnaround_times) / len(turnaround_times)
    return {
        "Waiting Times": waiting_times,
        "Turnaround Times": turnaround_times,
        "Avg Waiting Time": round(avg_waiting, 2),
        "Avg Turnaround Time": round(avg_turnaround, 2)
    }


def sjf_scheduling(processes, burst_times):
    """Simulate Shortest Job First (Non-Preemptive) Scheduling"""
    sorted_indices = np.argsort(burst_times)
    burst_sorted = [burst_times[i] for i in sorted_indices]
    waiting_times = [0]
    for i in range(1, len(burst_sorted)):
        waiting_times.append(waiting_times[-1] + burst_sorted[i - 1])
    avg_waiting = sum(waiting_times) / len(waiting_times)
    turnaround_times = [wt + bt for wt, bt in zip(waiting_times, burst_sorted)]
    avg_turnaround = sum(turnaround_times) / len(turnaround_times)
    return {
        "Order": [processes[i] for i in sorted_indices],
        "Waiting Times": waiting_times,
        "Turnaround Times": turnaround_times,
        "Avg Waiting Time": round(avg_waiting, 2),
        "Avg Turnaround Time": round(avg_turnaround, 2)
    }


def round_robin_scheduling(processes, burst_times, quantum):
    """Simulate Round Robin Scheduling"""
    remaining_times = burst_times[:]
    waiting_times = [0] * len(burst_times)
    turnaround_times = [0] * len(burst_times)
    t = 0
    done = False

    while not done:
        done = True
        for i in range(len(burst_times)):
            if remaining_times[i] > 0:
                done = False
                if remaining_times[i] > quantum:
                    t += quantum
                    remaining_times[i] -= quantum
                else:
                    t += remaining_times[i]
                    waiting_times[i] = t - burst_times[i]
                    remaining_times[i] = 0

    for i in range(len(burst_times)):
        turnaround_times[i] = burst_times[i] + waiting_times[i]

    avg_waiting = sum(waiting_times) / len(waiting_times)
    avg_turnaround = sum(turnaround_times) / len(turnaround_times)
    return {
        "Waiting Times": waiting_times,
        "Turnaround Times": turnaround_times,
        "Avg Waiting Time": round(avg_waiting, 2),
        "Avg Turnaround Time": round(avg_turnaround, 2)
    }


# ================================================================
# 🧩 WRAPPER FUNCTIONS
# ================================================================

def simulate_page_replacement(algo_name, pages, frames):
    """Run the selected Page Replacement Algorithm"""
    algo_name = algo_name.lower()
    if algo_name == "fifo":
        faults = fifo_page_replacement(pages, frames)
    elif algo_name == "lru":
        faults = lru_page_replacement(pages, frames)
    elif algo_name == "optimal":
        faults = optimal_page_replacement(pages, frames)
    else:
        raise ValueError("Unknown Page Replacement Algorithm")
    return {"Page Faults": faults, "Algorithm": algo_name.upper()}


def simulate_cpu_scheduling(algo_name, processes, burst_times, quantum=None):
    """Run the selected CPU Scheduling Algorithm"""
    algo_name = algo_name.lower()
    if algo_name == "fcfs":
        return fcfs_scheduling(processes, burst_times)
    elif algo_name == "sjf":
        return sjf_scheduling(processes, burst_times)
    elif algo_name in ["rr", "round robin", "roundrobin"]:
        if quantum is None:
            raise ValueError("Quantum required for Round Robin")
        return round_robin_scheduling(processes, burst_times, quantum)
    else:
        raise ValueError("Unknown Scheduling Algorithm")
=======
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
>>>>>>> 164492b38d85f20bd4b59d20d12867b6f15b82c7
