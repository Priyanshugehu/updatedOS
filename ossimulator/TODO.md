# TODO: Integrate Frontend with Trained Model

- [x] Create main.py as Flask server:
  - Load best_algo_model.pkl, scaler.pkl, label_encoder.pkl
  - Serve frontend files
  - Implement /predict endpoint to predict best algorithm
- [x] Edit frontend/pageReplacement/index.html:
  - Add locality input field (slider or number, 0.05-0.95)
- [x] Edit frontend/pageReplacement/script.js:
  - Modify simulate function to call /predict with derived features
  - Use predicted algorithm for simulation
  - Add Optimal algorithm implementation
  - Update UI to show predicted algorithm instead of manual select
- [x] Test the integration by running the server and simulating
