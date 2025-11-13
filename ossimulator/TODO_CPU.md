# TODO: Integrate CPU Scheduling Model

- [x] Update main.py:
  - Load best_cpu_scheduling_model.pkl, scaler_cpu.pkl, label_encoder_cpu.pkl
  - Add /predict_cpu endpoint to predict best CPU scheduling algorithm
- [x] Update frontend/scheduling/index.html:
  - Add inputs for arrival times and burst times (comma-separated)
- [x] Update frontend/scheduling/script.js:
  - Modify predictBtn to call /predict_cpu with inputs
  - Display predicted algorithm
  - Add simulateBtn to trigger simulation
  - Implement simulation functions for FCFS, SJF, Round Robin with Gantt chart, waiting/turnaround times
- [x] Update frontend/scheduling/style.css:
  - Add styles for Gantt chart
- [x] Test the CPU scheduling integration
