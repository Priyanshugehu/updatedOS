document.getElementById("predictBtn").addEventListener("click", async function () {
  const num_processes = parseInt(document.getElementById("num_processes").value);
  const avg_burst_time = parseFloat(document.getElementById("avg_burst_time").value);
  const avg_arrival_time = parseFloat(document.getElementById("avg_arrival_time").value);
  const avg_waiting_time = parseFloat(document.getElementById("avg_waiting_time").value);
  const cpu_utilization = parseFloat(document.getElementById("cpu_utilization").value);
  const time_quantum = parseFloat(document.getElementById("time_quantum").value);

  if (isNaN(num_processes) || num_processes <= 0 ||
      isNaN(avg_burst_time) || avg_burst_time <= 0 ||
      isNaN(avg_arrival_time) || avg_arrival_time < 0 ||
      isNaN(avg_waiting_time) || avg_waiting_time < 0 ||
      isNaN(cpu_utilization) || cpu_utilization < 0.5 || cpu_utilization > 1.0 ||
      isNaN(time_quantum) || time_quantum <= 0) {
    alert("Please enter valid inputs!");
    return;
  }

  // Call predict_cpu endpoint
  const response = await fetch('/predict_cpu', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      num_processes: num_processes,
      avg_burst_time: avg_burst_time,
      avg_arrival_time: avg_arrival_time,
      avg_waiting_time: avg_waiting_time,
      cpu_utilization: cpu_utilization,
      time_quantum: time_quantum
    })
  });
  const data = await response.json();
  const predicted_algo = data.predicted_algo;

  // Display predicted algorithm
  document.getElementById("result").innerHTML = `<h3>Predicted Best Algorithm: ${predicted_algo}</h3><p>Now simulate with this algorithm.</p>`;
});
