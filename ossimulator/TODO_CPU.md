# TODO: Integrate CPU Scheduling Model

- [x] Update main.py:
  - Load best_cpu_scheduling_model.pkl, scaler_cpu.pkl, label_encoder_cpu.pkl
  - Add /predict_cpu endpoint to predict best CPU scheduling algorithm
- [x] Update frontend/scheduling/script.js:
  - Modify predictBtn to call /predict_cpu with inputs
  - Display predicted algorithm
  - Simulate with the predicted algorithm
- [x] Test the CPU scheduling integration
