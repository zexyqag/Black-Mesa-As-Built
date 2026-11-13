"""What the downloads run (built with PyInstaller by .github/workflows/release.yml): bmab, by default bmab app, keeping its things
in the app's folder (data, downloads, cache, git, the redlines), with the settings the release was built with (settings.json in
the bundle: the redlines' address, the GitHub sign-in's client ID, its version) where the environment doesn't say otherwise. Without a console
(the Windows download) what bmab says goes to bmab.log in the app's folder."""
import json
import multiprocessing
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()  # the app's own processes (a job, the views' builds) start here as well
    if sys.platform == "linux":  # the bundle's libraries come first for the program itself only: git and the browser need the system's
        original = os.environ.pop("LD_LIBRARY_PATH_ORIG", None)
        if original is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = original
    home = Path(sys.executable).parent
    os.environ.setdefault("BMAB_HOME", str(home))
    os.environ.setdefault("BMAB_REDLINES", str(home / "redlines"))
    for name, value in json.loads((Path(sys._MEIPASS) / "settings.json").read_text(encoding="utf-8")).items():
        os.environ.setdefault(name, value)
    if sys.stdout is None:
        sys.stdout = sys.stderr = open(home / "bmab.log", "w", encoding="utf-8", buffering=1)
    if len(sys.argv) == 1:
        sys.argv.append("app")
    from bmab.__main__ import main
    main()
