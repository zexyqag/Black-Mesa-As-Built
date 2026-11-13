// What taking in others' changes asks of you, for the app's pages. showConflicts(el, list, base, done): what someone else changed
// that you changed too, each entry as yours and as theirs (only what differs), and which to keep; list from /conflicts (the page's
// own), base the app's address from this page, done() once one is decided. showBroughtBack(el, list, base, done): register entries
// that came back because something still names them; list from /brought-back, done() once seen.
function showConflicts(el, list, base, done) {
  const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  el.style.display = list.length ? "" : "none";
  if (!list.length) { el.innerHTML = ""; return; }
  const shown = v => v === undefined ? "none" : JSON.stringify(v);
  const differ = (a, b) => [...new Set([...Object.keys(a || {}), ...Object.keys(b || {})])].filter(k => shown((a || {})[k]) !== shown((b || {})[k]));
  const said = (e, keys) => e === null ? "deleted" : typeof e === "string" ? e.slice(0, 300) : keys.map(k => `${k}: ${shown(e[k])}`).join(", ");
  const what = c => { const e = c.yours || c.theirs || c.base; const kind = e && e.kind ? e.kind.replace("_", " ") : "claim";
    return c.unit.startsWith("maps/") ? (e && c.id === e.kind ? kind : `${kind} ${c.id}`) : `${c.unit.replace(/\.jsonl$/, "")}: ${c.id}`; };
  el.innerHTML = `<b>Someone else changed ${list.length === 1 ? "this" : `these ${list.length}`} too.</b> Keep yours or take theirs:` + list.map((c, i) => {
    const keys = differ(c.yours, c.theirs);
    return `<div style="margin:6px 0">${esc(what(c))}<br>yours: ${esc(said(c.yours, keys))}<br>theirs: ${esc(said(c.theirs, keys))}<br>` +
      `<button data-i="${i}" data-keep="yours">Keep yours</button> <button data-i="${i}" data-keep="theirs">Take theirs</button></div>`;
  }).join("");
  el.querySelectorAll("button").forEach(b => b.onclick = async () => {
    const c = list[b.dataset.i];
    const r = await fetch(base + "conflicts/decide", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ unit: c.unit, id: c.id, keep: b.dataset.keep }) });
    if (!r.ok) { alert(await r.text()); return; }
    done();
  });
}

function showBroughtBack(el, list, base, done) {
  const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  el.style.display = list.length ? "" : "none";
  if (!list.length) { el.innerHTML = ""; return; }
  const kind = u => ({ "places.jsonl": "place", "sources.jsonl": "source", "facts.jsonl": "fact", "events.jsonl": "event", evidence: "evidence" })[u] || u;
  el.innerHTML = `<b>Someone deleted ${list.length === 1 ? "this" : "these"}, but something still names ${list.length === 1 ? "it" : "them"}, so ${list.length === 1 ? "it is" : "they are"} back.</b> To remove one for good, first take it out of what names it.` +
    list.map(b => `<div>${esc(kind(b.unit))} ${esc(b.id)}: named by ${b.by.map(([u, i]) => esc(kind(u) + " " + i)).join(", ")}</div>`).join("") + `<button>Seen</button>`;
  el.querySelector("button").onclick = async () => {
    const r = await fetch(base + "brought-back/seen", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (!r.ok) { alert(await r.text()); return; }
    done();
  };
}
