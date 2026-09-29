"""Small shared helpers: .env loading, terminal color, and alert notifications."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import cv2

from gavi.backends import Verdict

# --- terminal color (auto-disabled when output isn't a TTY) ------------------
_USE_COLOR = sys.stdout.isatty()


def c(text: str, codes: str) -> str:
    return f"\033[{codes}m{text}\033[0m" if _USE_COLOR else text


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency): KEY=value lines; existing env wins."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def notify(rule: str, verdict: Verdict, frame, evidence_dir: str) -> None:
    ts = datetime.now()
    os.makedirs(evidence_dir, exist_ok=True)
    path = os.path.join(evidence_dir, ts.strftime("%Y%m%d_%H%M%S_%f.jpg"))
    cv2.imwrite(path, frame)
    print(
        c(
            f"🚨 [{ts:%Y-%m-%d %H:%M:%S}] ALERT: {rule}  "
            f"({verdict.confidence:.0%}) — {verdict.reason}",
            "1;97;41",  # bold white on red background
        ),
        flush=True,
    )
    print(c(f"          evidence: {path}", "90"), flush=True)
    if sys.platform == "darwin":
        body = verdict.reason.replace('"', "'")[:200]
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{body}" with title "VLM Alert: {rule[:60]}"',
            ],
            check=False,
        )
