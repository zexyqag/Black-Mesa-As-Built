// What someone else changed that you changed too, for the app's pages: each entry as yours and as theirs (only what differs), and
// which to keep. showConflicts(el, list, base, done): list from /conflicts (the page's own), base the app's address from this page,
// done() once one is decided.
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
