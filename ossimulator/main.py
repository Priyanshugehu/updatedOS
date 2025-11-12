from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and preprocessing tools for page replacement
model = joblib.load('best_algo_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Load the trained model and preprocessing tools for CPU scheduling
model_cpu = joblib.load('best_cpu_scheduling_model.pkl')
scaler_cpu = joblib.load('scaler_cpu.pkl')
le_cpu = joblib.load('label_encoder_cpu.pkl')

@app.route('/')
def index():
    return send_from_directory('osync frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('osync frontend', path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    num_pages = data['num_pages']
    memory_size = data['memory_size']  # frames
    locality = data['locality']
    sequence_len = data['sequence_len']

    # Compute derived features
    frames_to_pages = memory_size / num_pages
    freq_var = 0.8 * (1 - locality)

    # Prepare input for model
    df_input = pd.DataFrame([[num_pages, memory_size, locality, sequence_len, frames_to_pages, freq_var]],
                            columns=["num_pages", "memory_size", "locality", "sequence_len", "frames_to_pages", "freq_var"])
    scaled = scaler.transform(df_input)
    pred = model.predict(scaled)
    predicted_algo = le.inverse_transform(pred)[0]

    return jsonify({'predicted_algo': predicted_algo})

@app.route('/predict_cpu', methods=['POST'])
def predict_cpu():
    data = request.get_json()
    num_processes = data['num_processes']
    avg_burst_time = data['avg_burst_time']
    avg_arrival_time = data['avg_arrival_time']
    avg_waiting_time = data['avg_waiting_time']
    cpu_utilization = data['cpu_utilization']
    time_quantum = data['time_quantum']

    # Compute derived features
    burst_to_quantum = avg_burst_time / (time_quantum + 1e-8)
    waiting_to_arrival = avg_waiting_time / (avg_arrival_time + 1e-5)

    # Define features (same order used during training)
    features = [
        "num_processes", "avg_burst_time", "avg_arrival_time",
        "avg_waiting_time", "cpu_utilization", "time_quantum",
        "burst_to_quantum", "waiting_to_arrival"
    ]

    # Prepare input for model
    df_input = pd.DataFrame([[num_processes, avg_burst_time, avg_arrival_time, avg_waiting_time, cpu_utilization, time_quantum, burst_to_quantum, waiting_to_arrival]],
                            columns=features)
    scaled = scaler_cpu.transform(df_input)
    pred = model_cpu.predict(scaled)
    predicted_algo = le_cpu.inverse_transform(pred)[0]

    return jsonify({'predicted_algo': predicted_algo})

if __name__ == '__main__':
    app.run(debug=True)
