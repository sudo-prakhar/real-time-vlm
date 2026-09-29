"""Video-source and frame helpers shared by the CLIs, the engine, and the web app."""

from __future__ import annotations

import sys
import time

import cv2


def open_source(source: str) -> cv2.VideoCapture:
    # "0"/"1" -> webcam index; anything else -> file path or RTSP/HTTP URL.
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source!r}")
    # Webcams need a few frames to warm up; this also catches all-black frames
    # (on macOS, almost always a missing Camera permission).
    if source.isdigit():
        brightness = 0.0
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                brightness = float(frame.mean())
                if brightness > 5:
                    break
            time.sleep(0.05)
        if brightness <= 5:
            print(
                "WARNING: camera frames are black. On macOS this is almost always a "
                "Camera permission issue.\n"
                "  Fix: System Settings → Privacy & Security → Camera → enable the app "
                "running Python\n"
                "       (Terminal / iTerm / VS Code), then FULLY quit and reopen it.\n"
                "  Or try a different camera: --source 1  (or 2).",
                file=sys.stderr,
            )
    return cap


def motion_fraction(prev_gray, gray) -> float:
    """Fraction of pixels that changed meaningfully between two frames."""
    return float((cv2.absdiff(prev_gray, gray) > 25).mean())


def downscale(frame, max_side: int):
    """Shrink so the longest side <= max_side (fewer vision tokens = faster VLM)."""
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    s = max_side / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
