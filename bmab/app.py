"""bmab as an app on this computer (python -m bmab app): bmab.serve on its port, opened at its front page, where the first run
happens (the redlines, the games to copy in). On Windows in a window of its own, over the WebView2 that comes with Windows
(pywebview), which keeps what the pages keep (their settings) in this person's local AppData; closing it, or Quit on the front
page, ends the app, asking first while a job runs. Links out of the app open in the browser, the app's own pages that open apart
(the place tree) in a window of their own. Elsewhere in the browser, until Quit on the front page. The port stays the same from
run to run, as the pages' settings belong to it."""
import os
import sys
import threading
import webbrowser
from pathlib import Path

from . import jobs, serve

TITLE = "Black Mesa As Built"
# pywebview sends every link that opens apart to the browser: the app's own go to a window of their own instead
APART = """document.addEventListener("click", e => {
  const a = e.target.closest && e.target.closest("a[target=_blank]");
  if (a && a.origin === location.origin) { e.preventDefault(); window.pywebview.api.apart(a.href); }
}, true);"""


def run(port):
    try:
        listening = serve.server(port)
    except OSError:
        return _tell(f"Port {port} is taken: bmab is running already, or another program has it. Close it, or start bmab with --port and another number.")
    url = f"http://localhost:{port}/"
    if sys.platform == "win32":
        _window(url, listening)
    else:
        webbrowser.open(url)
        try:
            listening.serve_forever()
        except KeyboardInterrupt:
            pass
    serve.end()


def _window(url, listening):
    import webview
    api = _Pages(url)
    main = _open(url, api, maximized=True)
    quitting = threading.Event()

    def serving():  # until Quit on the front page, which closes the window
        listening.serve_forever()
        quitting.set()
        main.destroy()

    def closing():  # False keeps the window open; Quit has asked already
        busy = jobs.running()
        return quitting.is_set() or busy is None or _ask(f"bmab is still {busy}: closing it stops that. Close anyway?")

    def closed():  # the app's other windows go with it
        for w in list(webview.windows):
            if w is not main:
                w.destroy()
    main.events.closing += closing
    main.events.closed += closed
    threading.Thread(target=serving, daemon=True).start()
    webview.start(private_mode=False, storage_path=str(Path(os.environ["LOCALAPPDATA"]) / "bmab" / "window"))


def _open(url, api, **how):
    import webview
    w = webview.create_window(TITLE, url, js_api=api, width=1280, height=860, **how)
    w.events.loaded += lambda window: window.run_js(APART)
    return w


class _Pages:
    """What the window's pages may ask of it."""

    def __init__(self, url):
        self._url = url

    def apart(self, url):
        if url.startswith(self._url):
            _open(url, self)
        else:
            webbrowser.open(url)


def _ask(text):
    """OK or Cancel, over the app's windows (on the thread that runs them, as pywebview asks whether to close on it)."""
    import ctypes
    return ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x1 | 0x30 | 0x2000 | 0x10000) == 1  # OK/Cancel, warning, the thread's windows, in front


def _tell(text):
    print(text, file=sys.stderr, flush=True)
    if sys.platform == "win32":  # started from the app's icon, with nowhere to print to
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x10)
    return 1
