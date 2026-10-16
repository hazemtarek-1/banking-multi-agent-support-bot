"""Record the required live Streamlit demo in a real browser.

Run from the project root after training/building RAG and configuring Groq or Ollama:
    python record_demo.py

The output is artifacts/demo_recording.webm. It records the actual running
Streamlit app, not a slide deck or a pre-rendered mockup.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "app.py"
OUT = ROOT / "artifacts" / "demo_recording.webm"
PORT = int(os.getenv("DEMO_PORT", "8501"))
URL = f"http://127.0.0.1:{PORT}"

MESSAGES = [
    "Why was I charged a fee for my card payment?",
    "My card is not working.",
    "How do I order a physical card?",
    "Can I book a support appointment on 2026-10-10?",
    "Which countries do you support?",
]


def wait_for_port(host: str, port: int, timeout: float = 45) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Streamlit did not start on {host}:{port}")


def main() -> None:
    if not (ROOT / "artifacts" / "classifier.pt").exists():
        raise SystemExit("Missing artifacts/classifier.pt. Run the training command first.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        wait_for_port("127.0.0.1", PORT)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                record_video_dir=str(OUT.parent),
                record_video_size={"width": 1440, "height": 900},
            )
            page = context.new_page()
            page.goto(URL, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(3000)

            for message in MESSAGES:
                box = page.get_by_placeholder("How can I help you?")
                box.fill(message)
                box.press("Enter")
                page.wait_for_timeout(7000)

            page.wait_for_timeout(3000)
            video_path = page.video.path()
            context.close()
            browser.close()
            # Playwright may use a generated name. Move it to the requested output.
            Path(video_path).replace(OUT)

        print(f"Saved live browser recording to: {OUT}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
