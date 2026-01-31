import React, { useState, useEffect } from "react";

const SENTENCES = [
  "The Quick Brown Fox Jumps Over The Lazy Dog",
  "Pack my bag with five dozen liquor jugs",
  "Python 3.11 is faster than Python 2.7",
  "My password is Password@123 do not share it"
];

function App() {
  const [index, setIndex] = useState(0);
  const [typed, setTyped] = useState("");
  const [allTyped, setAllTyped] = useState("");
  const [events, setEvents] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleKeyDown = (e) => {
    setEvents((p) => [...p, { key: e.key, type: "down", time: Date.now() }]);
  };

  const handleKeyUp = (e) => {
    setEvents((p) => [...p, { key: e.key, type: "up", time: Date.now() }]);
  };

  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  const std = (a) => {
    if (a.length < 2) return 25;
    const m = mean(a);
    return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
  };

  const nextSentence = () => {
    if (!typed.trim()) {
      alert("Please type the sentence");
      return;
    }

    setAllTyped((p) => p + " " + typed);

    if (index < 3) {
      setIndex((p) => p + 1);
      setTyped("");
      return;
    }

    setTimeout(analyzeTyping, 50);
  };

  const analyzeTyping = async () => {
    setLoading(true);

    try {
      const downs = events.filter((e) => e.type === "down");
      const ups = events.filter((e) => e.type === "up");

      if (downs.length < 150) {
        alert("Please type more naturally");
        setLoading(false);
        return;
      }

      const dwells = [];
      const flights = [];
      const digraphs = [];

      for (let u of ups) {
        const d = downs.find((x) => x.key === u.key && x.time < u.time);
        if (d) {
          const dwell = u.time - d.time;
          if (dwell > 20 && dwell < 2000) dwells.push(dwell);
        }
      }

      for (let i = 0; i < downs.length - 1; i++) {
        const flight = downs[i + 1].time - ups[i]?.time;
        const digraph = downs[i + 1].time - downs[i].time;
        if (flight > 5 && flight < 2000) flights.push(flight);
        if (digraph > 5 && digraph < 2000) digraphs.push(digraph);
      }

      const totalTime =
        (downs[downs.length - 1].time - downs[0].time) / 60000;

      const safeTime = Math.max(totalTime, 0.5);

      const payload = {
        ks_count: downs.length,
        ks_rate: downs.length / (safeTime * 60),
        dwell_mean: mean(dwells),
        dwell_std: std(dwells),
        flight_mean: mean(flights),
        flight_std: std(flights),
        digraph_mean: mean(digraphs),
        digraph_std: std(digraphs),
        backspace_rate:
          events.filter((e) => e.key === "Backspace").length / events.length,
        wps: allTyped.split(" ").length / safeTime,
        wpm: allTyped.length / 5 / safeTime
      };

      const res = await fetch("http://127.0.0.1:5000/predict_sequential", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      setResult(data.top3_predictions);
    } catch (e) {
      alert("Prediction failed. Please retype clearly.");
      console.error(e);
    }

    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: "auto", padding: 30 }}>
      <h2>🔐 TypeID – Keystroke Dynamics</h2>

      <div style={{ background: "#eee", padding: 15 }}>
        {SENTENCES[index]}
      </div>

      <textarea
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        disabled={loading}
        style={{ width: "100%", height: 120, marginTop: 10 }}
      />

      <p>
        <b>Progress:</b> {events.length} keys | {allTyped.length} chars
      </p>

      <button onClick={nextSentence} disabled={loading}>
        {index < 3 ? "Next Sentence" : "Analyze Typing"}
      </button>

      {result && (
        <>
          <h3>🎯 Top 3 Predictions</h3>
          {result.map((r, i) => (
            <div key={i}>
              #{r.rank} {r.user} – {r.confidence.toFixed(1)}%
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export default App;
