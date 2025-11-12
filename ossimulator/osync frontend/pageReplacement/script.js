const predictBtn = document.getElementById("predictBtn");

predictBtn.addEventListener("click", async () => {
  const num_pages = parseInt(document.getElementById("num_pages").value);
  const memory_size = parseInt(document.getElementById("memory_size").value);
  const locality = parseFloat(document.getElementById("locality").value);
  const sequence_len = parseInt(document.getElementById("sequence_len").value);
  const resultDiv = document.getElementById("result");

  if (isNaN(num_pages) || isNaN(memory_size) || isNaN(locality) || isNaN(sequence_len)) {
    alert("Please enter valid input.");
    return;
  }

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        num_pages,
        memory_size,
        locality,
        sequence_len,
      }),
    });

    if (!response.ok) {
      throw new Error('Prediction failed');
    }

    const data = await response.json();
    const predictedAlgo = data.predicted_algo;

    resultDiv.innerHTML = `
      <h3>Predicted Best Algorithm: ${predictedAlgo}</h3>
      <p>Now enter the actual page reference string below to simulate:</p>
      <div class="form-section">
        <label>Page Reference String:</label>
        <input type="text" id="pages" placeholder="7, 0, 1, 2, 0, 3, 0, 4" />
        <button id="simulateBtn">Simulate ${predictedAlgo}</button>
      </div>
      <div id="simulationResult" class="glass-sub"></div>
      <div id="table-container" class="glass-sub"></div>
    `;

    // Add event listener for simulate button
    document.getElementById("simulateBtn").addEventListener("click", () => {
      const pages = document.getElementById("pages").value.split(",").map(x => parseInt(x.trim()));
      const frames = memory_size;
      const simulationResultDiv = document.getElementById("simulationResult");
      const tableContainer = document.getElementById("table-container");

      if (pages.some(isNaN)) {
        alert("Please enter valid page reference string.");
        return;
      }

      let frameHistory = [];
      let faults = 0;
      let hits = 0;

      if (predictedAlgo === "FIFO") {
        let frame = [];
        for (let p of pages) {
          if (frame.includes(p)) {
            hits++;
          } else {
            faults++;
            if (frame.length >= frames) frame.shift();
            frame.push(p);
          }
          frameHistory.push([...frame]);
        }
      } else if (predictedAlgo === "LRU") {
        let frame = [];
        let recent = new Map();
        for (let i = 0; i < pages.length; i++) {
          const p = pages[i];
          if (frame.includes(p)) {
            hits++;
          } else {
            faults++;
            if (frame.length < frames) {
              frame.push(p);
            } else {
              // find least recently used
              const lruPage = [...recent.entries()].reduce((a, b) => a[1] < b[1] ? a : b)[0];
              frame[frame.indexOf(lruPage)] = p;
              recent.delete(lruPage);
            }
          }
          recent.set(p, i);
          frameHistory.push([...frame]);
        }
      } else if (predictedAlgo === "Optimal") {
        let frame = [];
        for (let i = 0; i < pages.length; i++) {
          const p = pages[i];
          if (frame.includes(p)) {
            hits++;
          } else {
            faults++;
            if (frame.length < frames) {
              frame.push(p);
            } else {
              // find optimal page to replace
              let future = pages.slice(i + 1);
              let indices = [];
              for (let x of frame) {
                if (future.includes(x)) {
                  indices.push(future.indexOf(x));
                } else {
                  indices.push(Infinity);
                }
              }
              let maxIndex = indices.indexOf(Math.max(...indices));
              frame[maxIndex] = p;
            }
          }
          frameHistory.push([...frame]);
        }
      }

      simulationResultDiv.innerHTML = `<h3>Simulation Results for ${predictedAlgo}</h3>
        <p>Total Page Faults: ${faults}</p>
        <p>Total Hits: ${hits}</p>`;

      // Create table
      let tableHTML = `<table><tr><th>Step</th>`;
      for (let i = 1; i <= frames; i++) tableHTML += `<th>Frame ${i}</th>`;
      tableHTML += `</tr>`;

      frameHistory.forEach((state, idx) => {
        tableHTML += `<tr><td>${idx + 1}</td>`;
        for (let j = 0; j < frames; j++) {
          tableHTML += `<td>${state[j] !== undefined ? state[j] : "-"}</td>`;
        }
        tableHTML += `</tr>`;
      });
      tableHTML += `</table>`;
      tableContainer.innerHTML = tableHTML;
    });

  } catch (error) {
    console.error('Error:', error);
    resultDiv.innerHTML = '<p>Error predicting algorithm. Please try again.</p>';
  }
});
