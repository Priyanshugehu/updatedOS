document.getElementById("simulateBtn").addEventListener("click", function () {
  const algo = document.getElementById("algo").value;
  const pages = document.getElementById("pages").value.split(",").map(x => parseInt(x.trim()));
  const frames = parseInt(document.getElementById("frames").value);
  if (isNaN(frames) || pages.some(isNaN)) {
    alert("Please enter valid input.");
    return;
  }

  let resultDiv = document.getElementById("result");
  let frameHistory = [];
  let faults = 0;
  let hits = 0;

  if (algo === "FIFO") {
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
  } else if (algo === "LRU") {
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
  } else if (algo === "LFU") {
    let frame = [];
    let freq = {};
    let arrival = {};
    for (let i = 0; i < pages.length; i++) {
      const p = pages[i];
      if (frame.includes(p)) {
        hits++;
        freq[p]++;
      } else {
        faults++;
        if (frame.length < frames) {
          frame.push(p);
          freq[p] = 1;
          arrival[p] = i;
        } else {
          let minFreq = Math.min(...frame.map(x => freq[x]));
          let candidates = frame.filter(x => freq[x] === minFreq);
          let pageToRemove = candidates.reduce((a, b) => arrival[a] < arrival[b] ? a : b);
          frame.splice(frame.indexOf(pageToRemove), 1);
          delete freq[pageToRemove];
          delete arrival[pageToRemove];
          frame.push(p);
          freq[p] = 1;
          arrival[p] = i;
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
