document.getElementById("simulateBtn").addEventListener("click", async function () {
  const pagesInput = document.getElementById("pages").value.split(",").map(x => parseInt(x.trim()));
  const frames = parseInt(document.getElementById("frames").value);
  const locality = parseFloat(document.getElementById("locality").value);
  if (isNaN(frames) || pagesInput.some(isNaN) || isNaN(locality) || locality < 0.05 || locality > 0.95) {
    alert("Please enter valid input.");
    return;
  }

  // Derive features
  const num_pages = new Set(pagesInput).size;
  const sequence_len = pagesInput.length;
  const memory_size = frames;

  // Call predict endpoint
  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      num_pages: num_pages,
      memory_size: memory_size,
      locality: locality,
      sequence_len: sequence_len
    })
  });
  const data = await response.json();
  const algo = data.predicted_algo;

  // Display predicted algorithm
  document.getElementById("predictedAlgo").innerHTML = `Predicted Best Algorithm: ${algo}`;

  let resultDiv = document.getElementById("result");
  let frameHistory = [];
  let faults = 0;
  let hits = 0;

  if (algo === "FIFO") {
    let frame = [];
    for (let p of pagesInput) {
      if (frame.includes(p)) {
        hits++;
      } else {
        faults++;
        if (frame.length >= frames) frame.shift();
        frame.push(p);
      }
      frameHistory.push([...frame]);
    }
  } else if (algo === "LRU") {
    let frame = [];
    let recent = new Map();
    for (let i = 0; i < pagesInput.length; i++) {
      const p = pagesInput[i];
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
  } else if (algo === "Optimal") {
    // Optimal: replace the page that will not be used for the longest time in the future
    let frame = [];
    for (let i = 0; i < pagesInput.length; i++) {
      const p = pagesInput[i];
      if (frame.includes(p)) {
        hits++;
      } else {
        faults++;
        if (frame.length < frames) {
          frame.push(p);
        } else {
          // Find the page that will be used farthest in the future
          let farthest = -1;
          let pageToReplace = -1;
          for (let j = 0; j < frame.length; j++) {
            let nextUse = pagesInput.slice(i + 1).indexOf(frame[j]);
            if (nextUse === -1) {
              pageToReplace = j;
              break;
            } else if (nextUse > farthest) {
              farthest = nextUse;
              pageToReplace = j;
            }
          }
          frame[pageToReplace] = p;
        }
      }
      frameHistory.push([...frame]);
    }
  }

  resultDiv.innerHTML = `<h3>Results</h3>
    <p>Total Page Faults: ${faults}</p>
    <p>Total Hits: ${hits}</p>`;

  // Create table
  let tableContainer = document.getElementById("table-container");
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
