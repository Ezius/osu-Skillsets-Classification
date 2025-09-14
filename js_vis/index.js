const skillsets = ['Aim', 'Stream', 'Alt', 'Tech', 'Speed', 'Rhythm'];
const skillsetColors = ["#0059ce", "#ff9900", "#27ac1b", "#e02f2f", "#8f37ca", "#834716"];


let skillsetProbData = [{
    x: [1,1,1,1,1,1],
    y: skillsets,
    orientation: "h",
    type: "bar",
    marker: { color: skillsetColors, opacity: 1 }
}];

let layout = {
    title: "Skillset Probability",
    yaxis: { autorange: "reversed" }  // flip order
};

Plotly.newPlot("skillsetProbability", skillsetProbData, {
    title: "Skillset Probabilities",
    yaxis: { autorange: "reversed" },
    xaxis: { autorange: "reversed" }
});
// ensure placeholders exist once
Plotly.newPlot("skillsetspreadLine", [], { title: "Skillset Spread (Line)" });
Plotly.newPlot("skillsetspreadStacked", [], { title: "Skillset Spread (Stacked Area)" });

async function fetchData() {
  try {
    const res = await fetch("http://localhost:7272/skillsets.json");
    const data = await res.json();

    if (!data.labels || !data.skillvaluespread || !data.time) {
      console.warn("Missing keys in JSON:", Object.keys(data));
      return;
    }

    const labels = data.labels;
    const time = data.time;
    const sv = data.skillvaluespread;

    Plotly.update("skillsetProbability", {
        x: [data.label_probability],
        y: [data.labels]
    }, {
        font: { size: 15 },
        height:647,
        margin: { l: 80, r: 6, t: 60, b: 25 }
    });

    title = data.title + ' [' + data.difficulty + '] ' + data.id

    document.getElementById("pageTitle").innerText = `${data.title} [${data.difficulty}] ${data.id}`;

    // DEBUG: log incoming shape
    console.log("Incoming skillvaluespread shape:", "outer =", sv.length, "inner =", Array.isArray(sv[0]) ? sv[0].length : "scalar");

    // Normalize into per-skill arrays: perSkill[i] = Array over time
    let perSkill;
    // Case A: already 6 x n (per-skill arrays)
    if (sv.length === labels.length && Array.isArray(sv[0]) && sv[0].length === time.length) {
      perSkill = sv;
      console.log("Detected shape: per-skill (6 x n)");
    // Case B: n x 6 (time-major) -> transpose
    } else if (Array.isArray(sv[0]) && sv[0].length === labels.length && sv.length === time.length) {
      perSkill = labels.map((_, col) => sv.map(row => row[col]));
      console.log("Detected shape: time-major (n x 6), transposed to per-skill");
    // Case C: other ambiguous shapes — try to transpose if inner length equals labels.length
    } else if (Array.isArray(sv[0]) && sv[0].length === labels.length) {
      perSkill = labels.map((_, col) => sv.map(row => row[col]));
      console.log("Heuristic transpose applied (inner length == labels.length)");
    } else {
      console.error("Unrecognized skillvaluespread shape — cannot plot reliably.");
      return;
    }

    // Safety: ensure perSkill[i].length matches time.length (or adjust)
    const n = time.length;
    perSkill = perSkill.map(arr => {
      if (!Array.isArray(arr)) return Array(n).fill(0);
      if (arr.length === n) return arr;
      // If lengths mismatch, try trimming or padding with zeros
      if (arr.length > n) return arr.slice(0, n);
      // arr.length < n -> pad with last value or zeros
      const padded = arr.slice();
      while (padded.length < n) padded.push(padded.length ? padded[padded.length-1] : 0);
      return padded;
    });

    // Build line traces (one trace per skill)
    const lineTraces = perSkill.map((values, i) => ({
      x: time,
      y: values,
      mode: "lines",
      name: labels[i],
      line: { color: skillsetColors[i % skillsetColors.length], width: 2 },
      hoverinfo: "name+x+y"
    }));

    Plotly.react("skillsetspreadLine", lineTraces, {
      title: "Skillset Spread (Line)",
      paper_bgcolor: "#ffffffff",
      plot_bgcolor: "#ffffffff",
      height:321,
      yaxis: { range: [0, 1], showticklabels: false },
      xaxis: { showticklabels: false },
      margin: { l: 6, r: 6, t: 45, b: 10 },
      showlegend:false
    });

    // Build stacked area traces
    const stackedTraces = perSkill.map((values, i) => ({
      x: time,
      y: values,
      name: labels[i],
      stackgroup: "one",
      mode: "none",
      fillcolor: skillsetColors[i % skillsetColors.length],
      hoverinfo: "skip"
    }));

    Plotly.react("skillsetspreadStacked", stackedTraces, {
      title: "Skillset Spread (Stacked Area)",
      paper_bgcolor: "#ffffffff",
      plot_bgcolor: "#ffffffff",
      height:321,
      yaxis: { range: [0, 1], showticklabels: false},
      xaxis: { showticklabels: true, nticks: 20 },
      font:{size:15},
      margin: { l: 6, r: 6, t: 45, b: 25 },
      showlegend:false
    });

  } catch (e) {
    console.error("Failed to fetch/plot data", e);
  }
}

// initial fetch + periodic updates
fetchData();
setInterval(fetchData, 400);
