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
  document.getElementById("simulateBtn").style.display = "inline-block";
  document.getElementById("simulateBtn").onclick = () => simulateScheduling(predicted_algo);
});

function simulateScheduling(algo) {
  const arrivalInput = document.getElementById("arrival_times").value.split(",").map(x => parseFloat(x.trim()));
  const burstInput = document.getElementById("burst_times").value.split(",").map(x => parseFloat(x.trim()));
  const quantum = parseFloat(document.getElementById("time_quantum").value);

  if (arrivalInput.length !== burstInput.length || arrivalInput.some(isNaN) || burstInput.some(isNaN)) {
    alert("Please enter valid arrival and burst times, same number for each.");
    return;
  }

  const processes = arrivalInput.map((at, i) => ({ id: i + 1, arrival: at, burst: burstInput[i], remaining: burstInput[i] }));

  let resultDiv = document.getElementById("result");
  let ganttChart = [];
  let waitingTimes = new Array(processes.length).fill(0);
  let turnaroundTimes = new Array(processes.length).fill(0);
  let completionTimes = new Array(processes.length).fill(0);

  if (algo === "FCFS") {
    // FCFS: First Come First Served
    processes.sort((a, b) => a.arrival - b.arrival);
    let currentTime = 0;
    processes.forEach(p => {
      if (currentTime < p.arrival) currentTime = p.arrival;
      waitingTimes[p.id - 1] = currentTime - p.arrival;
      ganttChart.push({ process: p.id, start: currentTime, end: currentTime + p.burst });
      currentTime += p.burst;
      completionTimes[p.id - 1] = currentTime;
      turnaroundTimes[p.id - 1] = completionTimes[p.id - 1] - p.arrival;
    });
  } else if (algo === "SJF") {
    // SJF: Shortest Job First (non-preemptive)
    let readyQueue = [];
    let currentTime = 0;
    let completed = 0;
    let index = 0;
    while (completed < processes.length) {
      // Add arrived processes to queue
      while (index < processes.length && processes[index].arrival <= currentTime) {
        readyQueue.push(processes[index]);
        index++;
      }
      if (readyQueue.length === 0) {
        currentTime = processes[index].arrival;
        continue;
      }
      // Select shortest burst
      readyQueue.sort((a, b) => a.burst - b.burst);
      let p = readyQueue.shift();
      waitingTimes[p.id - 1] = currentTime - p.arrival;
      ganttChart.push({ process: p.id, start: currentTime, end: currentTime + p.burst });
      currentTime += p.burst;
      completionTimes[p.id - 1] = currentTime;
      turnaroundTimes[p.id - 1] = completionTimes[p.id - 1] - p.arrival;
      completed++;
    }
  } else if (algo === "Round Robin") {
    // Round Robin
    let queue = [];
    let currentTime = 0;
    let index = 0;
    while (queue.length > 0 || index < processes.length) {
      // Add arrived processes
      while (index < processes.length && processes[index].arrival <= currentTime) {
        queue.push(processes[index]);
        index++;
      }
      if (queue.length === 0) {
        currentTime = processes[index].arrival;
        continue;
      }
      let p = queue.shift();
      let execTime = Math.min(quantum, p.remaining);
      ganttChart.push({ process: p.id, start: currentTime, end: currentTime + execTime });
      currentTime += execTime;
      p.remaining -= execTime;
      if (p.remaining > 0) {
        queue.push(p);
      } else {
        completionTimes[p.id - 1] = currentTime;
        turnaroundTimes[p.id - 1] = completionTimes[p.id - 1] - p.arrival;
        waitingTimes[p.id - 1] = turnaroundTimes[p.id - 1] - p.burst;
      }
    }
  }

  // Display results
  let html = `<h3>Simulation Results for ${algo}</h3>`;
  html += `<h4>Gantt Chart:</h4><div class="gantt">`;
  ganttChart.forEach(block => {
    html += `<div class="gantt-block" style="width: ${ (block.end - block.start) * 20 }px;">P${block.process}<br>${block.start}-${block.end}</div>`;
  });
  html += `</div>`;
  html += `<h4>Process Details:</h4><table><tr><th>Process</th><th>Arrival</th><th>Burst</th><th>Waiting</th><th>Turnaround</th></tr>`;
  processes.forEach((p, i) => {
    html += `<tr><td>P${p.id}</td><td>${p.arrival}</td><td>${p.burst}</td><td>${waitingTimes[i]}</td><td>${turnaroundTimes[i]}</td></tr>`;
  });
  html += `</table>`;
  html += `<p>Average Waiting Time: ${(waitingTimes.reduce((a,b)=>a+b,0)/waitingTimes.length).toFixed(2)}</p>`;
  html += `<p>Average Turnaround Time: ${(turnaroundTimes.reduce((a,b)=>a+b,0)/turnaroundTimes.length).toFixed(2)}</p>`;
  resultDiv.innerHTML = html;
}
