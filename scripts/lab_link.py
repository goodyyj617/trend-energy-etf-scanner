"""Open the research lab from a browser bookmark (Windows).

    python scripts/lab_link.py register     # once: lets trendlab:// links start the lab
    python scripts/lab_link.py unregister   # undo
    python scripts/lab_link.py open trendlab://saved

`register` adds the per-user link type trendlab:// (HKCU\\Software\\Classes\\trendlab, no admin
rights). A Chrome bookmark to trendlab://saved then runs `open`: it starts the lab in a minimized
window if it is not running yet, waits until it answers, and opens the page in the browser.
Closing the minimized "Trend Lab" window stops the lab.
Before starting the lab it brings merged code from GitHub (scripts/lab_update.py).
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEME = "trendlab"
HOST, PORT = "127.0.0.1", 8501
# trendlab://<name> -> page url path (st.navigation derives it from the page function name)
PAGES = {"": "", "research": "", "holdout": "holdout_page", "saved": "saved_page", "glossary": "glossary_page"}


def listening() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def start_lab() -> None:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 7  # SW_SHOWMINNOACTIVE
    subprocess.Popen(
        ["cmd", "/c", "title Trend Lab & " + subprocess.list2cmdline(
            [str(python), "-m", "streamlit", "run", str(ROOT / "lab" / "app.py"),
             "--server.headless", "true", "--server.port", str(PORT)])],
        cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)


def open_link(link: str) -> None:
    name = link.split(":", 1)[-1].strip("/").lower()
    path = PAGES.get(name, "")
    if not listening():
        import lab_update  # scripts/ is on sys.path when this file runs as a script
        lab_update.main([str(ROOT)])  # first bring merged code from GitHub; never raises, never blocks the start
        start_lab()
        deadline = time.time() + 90
        while not listening() and time.time() < deadline:
            time.sleep(0.5)
        time.sleep(1.0)  # let the app finish its first load
    webbrowser.open(f"http://{HOST}:{PORT}/{path}")


def register() -> None:
    import winreg
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    command = f'"{pythonw}" "{Path(__file__).resolve()}" open "%1"'
    base = rf"Software\Classes\{SCHEME}"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "URL:Trend Lab")
        winreg.SetValueEx(k, "URL Protocol", 0, winreg.REG_SZ, "")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\shell\open\command") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, command)
    print(f"Registered {SCHEME}:// -> {command}")


def unregister() -> None:
    import winreg
    base = rf"Software\Classes\{SCHEME}"
    for sub in (r"\shell\open\command", r"\shell\open", r"\shell", ""):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, base + sub)
        except FileNotFoundError:
            pass
    print(f"Removed {SCHEME}://")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["register"]:
        register()
    elif args[:1] == ["unregister"]:
        unregister()
    elif args[:1] == ["open"]:
        open_link(args[1] if len(args) > 1 else "")
    else:
        raise SystemExit(__doc__)
