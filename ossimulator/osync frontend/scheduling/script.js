const algoSelect = document.getElementById("algo");
const quantumField = document.getElementById("quantumField");
const simulateBtn = document.getElementById("simulateBtn");

algoSelect.addEventListener("change", () => {
  if (algoSelect.value === "Round Robin") {
    quantumField.style.display = "block";
  } else {
    quantumField.style.display = "none";
  }
});

simulateBtn.addEventListener("click", () => {
  const algo = algoSelect.value;
  const burstInput = document.getElementById("burst").value;
  const bursts = burstInput.split(",").map(x => parseInt(x.trim())).filter(x => !isNaN(x));
  const quantum = parseInt(document.getElementById("quantum").value);
  const resultDiv = document.getElementById("result");
  const tableContainer = document.getElementById("table-container");

  if (bursts.length === 0) {
    alert("Please enter valid burst times!");
    return;
  }

  let n = bursts.length;
  let waiting = new Array(n).fill(0);
  let turnaround = new Array(n).fill(0);
  let timeline = [];

  if (algo === "FCFS") {
    let current = 0;
    for (let i = 0; i < n; i++) {
      waiting[i] = current;
      current += bursts[i];
      turnaround[i] = current;
      timeline.push({ process: `P${i+1}`, start: waiting[i], end: turnaround[i] });
    }
  }

  else if (algo === "SJF") {
    let processes = bursts.map((b, i) => ({ id: i, burst: b }));
    processes.sort((a, b) => a.burst - b.burst);
    let current = 0;
    for (let p of processes) {
      waiting[p.id] = current;
      current += p.burst;
      turnaround[p.id] = current;
      timeline.push({ process: `P${p.id+1}`, start: waiting[p.id], end: turnaround[p.id] });
    }
  }

  else if (algo === "Round Robin") {
    if (isNaN(quantum) || quantum <= 0) {
      alert("Enter a valid quantum!");
      return;
    }
    let remaining = [...bursts];
    let time = 0;
    let completed = 0;
    let queue = [];

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

  resultDiv.innerHTML = `
    <h3>Results</h3>
    <p>Average Waiting Time: ${avgWait}</p>
    <p>Average Turnaround Time: ${avgTurn}</p>
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

  // Optional: timeline console log
  console.log("Timeline:", timeline);
});
