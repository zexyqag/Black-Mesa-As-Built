"""Find installed Steam apps and Workshop items through Steam's own library list."""
import os
import re
from pathlib import Path

_KV = re.compile(r'"([^"]*)"\s+"([^"]*)"')
LINUX_ROOTS = ["~/.steam/steam", "~/.local/share/Steam", "~/.var/app/com.valvesoftware.Steam/.local/share/Steam",  # native, Flatpak
               "~/snap/steam/common/.local/share/Steam", "/mnt/c/Program Files (x86)/Steam"]  # Snap, and Windows' Steam seen from WSL


def _windows_roots():
    """Where Steam says it is installed (the registry), and its usual folder."""
    import winreg
    found = []
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            found.append(winreg.QueryValueEx(k, "SteamPath")[0])
    except OSError:
        pass
    return [*found, os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Steam")]


def _native(p):
    """A path from Steam's files (backslashes escaped) as this system reaches it: Windows' E:\\Steam is /mnt/e/Steam under WSL."""
    p = p.replace("\\\\", "\\")
    m = re.match(r"^([A-Za-z]):[\\/]+(.*)$", p)
    if m and os.name == "posix":
        return Path("/mnt") / m.group(1).lower() / m.group(2).replace("\\", "/")
    return Path(p)


def roots():
    """The Steam installs found here; BMAB_STEAM (os.pathsep-separated) names them instead."""
    env = os.environ.get("BMAB_STEAM")
    found = [Path(r).expanduser() for r in (env.split(os.pathsep) if env else _windows_roots() if os.name == "nt" else LINUX_ROOTS)]
    return list({r.resolve(): r for r in found if (r / "steamapps").is_dir()}.values())


def libraries():
    libs = []
    for r in roots():
        vdf = r / "steamapps" / "libraryfolders.vdf"
        paths = [_native(v) for k, v in _KV.findall(vdf.read_text("utf-8", "replace")) if k == "path"] if vdf.exists() else []
        for p in [r, *paths]:
            if (p / "steamapps").is_dir() and p.resolve() not in [l.resolve() for l in libs]:
                libs.append(p)
    return libs


def app_dir(app):
    for lib in libraries():
        acf = lib / "steamapps" / f"appmanifest_{app}.acf"
        if acf.exists():
            kv = dict(_KV.findall(acf.read_text("utf-8", "replace")))
            d = lib / "steamapps" / "common" / kv["installdir"]
            if d.is_dir():
                return d
    return None


def workshop_dir(app, item):
    for lib in libraries():
        d = lib / "steamapps" / "workshop" / "content" / str(app) / str(item)
        if d.is_dir():
            return d
    return None
