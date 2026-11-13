// Signing in with GitHub, for the app's pages: GitHub's device flow, where the person types a code at GitHub and the app asks GitHub
// when they have. say(html) shows how it goes. Resolves to "done", or to what stopped it, in words.
async function gitHubSignIn(say) {
  const post = async (path, body) => {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  };
  const plain = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  let d;
  try { d = await post("/login/start", {}); } catch (e) { return `Signing in failed: ${e.message}`; }
  say(`Type <b>${plain(d.user_code)}</b> at <a href="${plain(d.verification_uri)}" target="_blank" rel="noopener">${plain(d.verification_uri)}</a> (copied, if your browser lets it be) and allow it. Waiting for GitHub`);
  try { await navigator.clipboard.writeText(d.user_code); } catch (e) { }
  window.open(d.verification_uri, "_blank", "noopener");
  let wait = (d.interval || 5) * 1000, said = "expired_token";
  const end = Date.now() + (d.expires_in || 900) * 1000;
  while (Date.now() < end) {
    await new Promise(r => setTimeout(r, wait));
    let r;
    try { r = await post("/login/poll", { device_code: d.device_code }); } catch (e) { said = e.message; break; }
    said = r.state;
    if (said === "refused") { said = r.message; break; }
    if (said === "slow_down") wait = (r.interval || wait / 1000 + 5) * 1000;
    else if (said !== "authorization_pending") break;
  }
  return said === "done" ? "done" : { access_denied: "You turned down signing in", expired_token: "The code ran out: sign in again" }[said] || `Signing in failed: ${said}`;
}
