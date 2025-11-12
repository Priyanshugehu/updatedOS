const predictBtn = document.getElementById("predictBtn");

predictBtn.addEventListener("click", async () => {
  const num_processes = parseInt(document.getElementById("num_processes").value);
  const avg_burst_time = parseFloat(document.getElementById("avg_burst_time").value);
  const avg_arrival_time = parseFloat(document.getElementById("avg_arrival_time").value);
  const avg_waiting_time = parseFloat(document.getElementById("avg_waiting_time").value);
  const cpu_utilization = parseFloat(document.getElementById("cpu_utilization").value);
  const time_quantum = parseFloat(document.getElementById("time_quantum").value);
  const resultDiv = document.getElementById("result");

  if (isNaN(num_processes) || num_processes <= 0) {
    alert("Please enter a valid number of processes!");
    return;
  }

  try {
    const response = await fetch('/predict_cpu', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        num_processes,
        avg_burst_time,
        avg_arrival_time,
        avg_waiting_time,
        cpu_utilization,
        time_quantum,
      }),
    });

    if (!response.ok) {
      throw new Error('Prediction failed');
    }

    const data = await response.json();
    const predictedAlgo = data.predicted_algo;

    resultDiv.innerHTML = `
      <h3>Predicted Best Algorithm: ${predictedAlgo}</h3>
      <p>Now enter the actual process details below to simulate:</p>
      <div class="form-section">
        <label>Enter Burst Times (comma separated):</label>
        <input type="text" id="burst" placeholder="5, 3, 8, 6" />
        ${predictedAlgo === 'RR' ? `
        <label>Time Quantum:</label>
        <input type="number" id="quantum" min="1" value="${time_quantum}" />
        ` : ''}
        <button id="simulateBtn">Simulate ${predictedAlgo}</button>
      </div>
      <div id="simulationResult" class="glass-sub"></div>
      <div id="table-container" class="glass-sub"></div>
    `;

    // Add event listener for simulate button
    document.getElementById("simulateBtn").addEventListener("click", () => {
      const burstInput = document.getElementById("burst").value;
      const bursts = burstInput.split(",").map(x => parseInt(x.trim())).filter(x => !isNaN(x));
      const quantum = predictedAlgo === 'RR' ? parseInt(document.getElementById("quantum").value) : null;
      const simulationResultDiv = document.getElementById("simulationResult");
      const tableContainer = document.getElementById("table-container");

      if (bursts.length === 0) {
        alert("Please enter valid burst times!");
        return;
      }

      let n = bursts.length;
      let waiting = new Array(n).fill(0);
      let turnaround = new Array(n).fill(0);
      let timeline = [];

      if (predictedAlgo === "FCFS") {
        let current = 0;
        for (let i = 0; i < n; i++) {
          waiting[i] = current;
          current += bursts[i];
          turnaround[i] = current;
          timeline.push({ process: `P${i+1}`, start: waiting[i], end: turnaround[i] });
        }
      } else if (predictedAlgo === "SJF") {
        let processes = bursts.map((b, i) => ({ id: i, burst: b }));
        processes.sort((a, b) => a.burst - b.burst);
        let current = 0;
        for (let p of processes) {
          waiting[p.id] = current;
          current += p.burst;
          turnaround[p.id] = current;
          timeline.push({ process: `P${p.id+1}`, start: waiting[p.id], end: turnaround[p.id] });
        }
      } else if (predictedAlgo === "RR") {
        if (isNaN(quantum) || quantum <= 0) {
          alert("Enter a valid quantum!");
          return;
        }
        let remaining = [...bursts];
        let time = 0;
        let completed = 0;

        while (completed < n) {
          for (let i = 0; i < n; i++) {
            if (remaining[i] > 0) {
              if (remaining[i] > quantum) {
                timeline.push({ process: `P${i+1}`, start: time, end: time + quantum });
                time += quantum;
                remaining[i] -= quantum;
              } else {
                timeline.push({ process: `P${i+1}`, start: time, end: time + remaining[i] });
                time += remaining[i];
                waiting[i] = time - bursts[i];
                remaining[i] = 0;
                completed++;
              }
              turnaround[i] = waiting[i] + bursts[i];
            }
          }
        }
      }

      let avgWait = (waiting.reduce((a,b)=>a+b,0)/n).toFixed(2);
      let avgTurn = (turnaround.reduce((a,b)=>a+b,0)/n).toFixed(2);

      simulationResultDiv.innerHTML = `
        <h3>Simulation Results for ${predictedAlgo}</h3>
        <p>Average Waiting Time: ${avgWait} ms</p>
        <p>Average Turnaround Time: ${avgTurn} ms</p>
      `;

      let tableHTML = `
        <table>
          <tr><th>Process</th><th>Burst Time</th><th>Waiting Time</th><th>Turnaround Time</th></tr>
      `;
      for (let i = 0; i < n; i++) {
        tableHTML += `<tr>
          <td>P${i + 1}</td>
          <td>${bursts[i]}</td>
          <td>${waiting[i]}</td>
          <td>${turnaround[i]}</td>
        </tr>`;
      }
      tableHTML += "</table>";
      tableContainer.innerHTML = tableHTML;

      console.log("Timeline:", timeline);
    });

  } catch (error) {
    console.error('Error:', error);
    resultDiv.innerHTML = '<p>Error predicting algorithm. Please try again.</p>';
  }
});
