#!/usr/bin/env python3
"""Launcher for the CBCT app.

Double-click `run_app.bat` (Windows), `run_app.command` (macOS), or run
`python launch.py`. It starts the Streamlit server on a free port and opens your
browser. Close the console window to stop the app.

This launcher uses the Python environment that has the app's dependencies
installed (streamlit, torch, nnunetv2, ...). If you build it into a .exe with
PyInstaller, the .exe still relies on that environment for inference.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


def app_base() -> Path:
    if getattr(sys, "frozen", False):          # running as a PyInstaller .exe
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_python() -> str | None:
    if not getattr(sys, "frozen", False):
        return sys.executable
    for name in ("python", "python3", "pythonw"):
        p = shutil.which(name)
        if p:
            return p
    return None


def free_port(preferred: int = 8501) -> int:
    for port in (preferred, 8502, 8503, 8504, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def wait_health(port: int, timeout: float = 90.0) -> bool:
    url = f"http://localhost:{port}/_stcore/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url, timeout=2).read().strip() == b"ok":
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    base = app_base()
    app = base / "app.py"
    if not app.exists():
        print(f"ERROR: app.py not found next to the launcher (looked in {base}).")
        input("Press Enter to close...")
        return 1

    py = find_python()
    if not py:
        print("ERROR: could not find a Python interpreter with the app's "
              "dependencies. Activate your environment and try again.")
        input("Press Enter to close...")
        return 1

    port = free_port()
    cmd = [py, "-m", "streamlit", "run", str(app),
           "--server.headless=true", f"--server.port={port}",
           "--browser.gatherUsageStats=false"]

    print("=" * 60)
    print(" CBCT Arch & Root Pipeline")
    print(" Starting server... (close this window to stop the app)")
    print("=" * 60)
    proc = subprocess.Popen(cmd, cwd=str(base))
    try:
        if wait_health(port):
            url = f"http://localhost:{port}"
            print(f"\n  App running at: {url}\n")
            try:
                webbrowser.open(url)
            except Exception:
                print("  (Open that URL in your browser manually.)")
        else:
            print("\nERROR: the server did not become ready. See messages above.")
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
