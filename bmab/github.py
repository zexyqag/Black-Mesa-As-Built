"""Signing in with GitHub through its device flow (no secret and no callback page: the person types a short code at GitHub), the
account it gives (kept in the person's own settings folder, readable by them alone, until they sign out), and what is done as
them: who they are, whether they may push to a repository, their fork of it and a pull request from it. Settings:
BMAB_GITHUB_CLIENT_ID, an OAuth app with Device Flow enabled; BMAB_GITHUB_SCOPE, what it asks for (repo by default; public_repo
is enough for public repositories).
"""
import json
import os
import re
import urllib.error
from pathlib import Path
import urllib.parse
import urllib.request


def client_id():
    return os.environ.get("BMAB_GITHUB_CLIENT_ID")


def _ask(request):
    """GitHub's answer; ValueError with its message when it refuses."""
    try:
        with urllib.request.urlopen(request, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            said = json.loads(e.read()).get("message", "")
        except ValueError:
            said = ""
        raise ValueError(f"GitHub said {e.code} {said or e.reason}") from None
    except urllib.error.URLError as e:
        raise ValueError(f"GitHub can't be reached: {e.reason}") from None


def _form(url, fields):
    return _ask(urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), headers={"Accept": "application/json"}))


def _api(path, token, method="GET", body=None):
    return _ask(urllib.request.Request("https://api.github.com" + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                       headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}))


def start():
    """A code for the person to enter: {device_code, user_code, verification_uri, interval, expires_in}."""
    got = _form("https://github.com/login/device/code", {"client_id": client_id(), "scope": os.environ.get("BMAB_GITHUB_SCOPE", "repo")})
    if "user_code" not in got:  # a client ID GitHub doesn't know, or one without Device Flow
        raise ValueError(f"GitHub won't sign in with this client ID: {got.get('error_description') or got.get('error', got)}")
    return got


def poll(device_code):
    """{access_token} once the person has entered the code, else {error}: authorization_pending, slow_down, expired_token or access_denied."""
    return _form("https://github.com/login/oauth/access_token",
                 {"client_id": client_id(), "device_code": device_code, "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})


def user(token):
    """{login, name, email}: the email is GitHub's no-reply address for them, so commits count as theirs."""
    u = _api("/user", token)
    return {"login": u["login"], "name": u.get("name") or u["login"], "email": f"{u['id']}+{u['login']}@users.noreply.github.com"}


def _kept():
    """The account's file in this person's settings folder: AppData on Windows, else $XDG_CONFIG_HOME or ~/.config."""
    return Path(os.environ.get("APPDATA" if os.name == "nt" else "XDG_CONFIG_HOME") or Path.home() / ".config") / "bmab" / "github.json"


def account():
    """The account signed in on this computer: {token, login, name, email}, or None."""
    try:
        return json.loads(_kept().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def remember(token):
    """Keep the account a token signs in as (readable by this user alone); it."""
    kept = {**user(token), "token": token}
    path = _kept()
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(json.dumps(kept))
    path.chmod(0o600)
    return kept


def forget():
    _kept().unlink(missing_ok=True)


def repository(address):
    """owner/name of a GitHub repository's git address (https or ssh); None for one elsewhere."""
    m = re.match(r"(?:https://(?:[^@/]+@)?github\.com/|git@github\.com:)([^/]+/[^/]+?)(?:\.git)?/?$", address.strip())
    return m.group(1) if m else None


def access(repo, token):
    """(whether the person may push to repo, its default branch)."""
    r = _api(f"/repos/{repo}", token)
    return bool(r.get("permissions", {}).get("push")), r["default_branch"]


def fork(repo, token):
    """The person's fork of repo, made if they have none: owner/name."""
    return _api(f"/repos/{repo}/forks", token, "POST", {})["full_name"]


def pull_request(repo, head, base, title, body, token):
    """The open pull request into repo from head (owner:branch), opened if there is none: its address."""
    open_ = _api(f"/repos/{repo}/pulls?state=open&head={urllib.parse.quote(head)}", token)
    if open_:
        return open_[0]["html_url"]
    return _api(f"/repos/{repo}/pulls", token, "POST", {"title": title, "head": head, "base": base, "body": body})["html_url"]
