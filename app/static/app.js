// Sends comments to /predict and renders one card per comment with a bar per label
const $ = (id) => document.getElementById(id);

$("threshold").addEventListener("input", (e) => ($("thVal").textContent = Number(e.target.value).toFixed(2)));
$("run").addEventListener("click", analyze);

async function analyze() {
  const comments = $("comments").value.split("\n").filter((c) => c.trim());
  if (!comments.length) return ($("status").textContent = "Enter at least one comment.");

  $("status").textContent = "Analyzing...";
  $("results").replaceChildren();
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments, threshold: Number($("threshold").value) }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    $("status").textContent = "";
    data.predictions.forEach((p) => $("results").appendChild(card(p)));
  } catch (err) {
    $("status").textContent = `Error: ${err.message}`;
  }
}

// Builds the card with textContent only, so user text can never inject HTML
function card(p) {
  const el = document.createElement("article");
  el.className = `card ${p.is_toxic ? "toxic" : "clean"}`;

  const head = document.createElement("div");
  head.className = "card-head";
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = p.is_toxic ? "Toxic" : "Clean";
  const text = document.createElement("p");
  text.className = "comment";
  text.textContent = p.comment;
  head.append(badge, text);
  el.appendChild(head);

  for (const [label, score] of Object.entries(p.scores)) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const name = document.createElement("span");
    name.textContent = window.LABEL_NAMES[label] || label;
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = `fill ${p.labels.includes(label) ? "hit" : ""}`;
    fill.style.width = `${Math.round(score * 100)}%`;
    track.appendChild(fill);
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${Math.round(score * 100)}%`;
    row.append(name, track, pct);
    el.appendChild(row);
  }
  return el;
}
