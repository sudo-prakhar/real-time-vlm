#!/usr/bin/env python3
"""Build the GAVI hero video + its real HUD tracks.

Three acts, one story:
  A. 3x3 CCTV wall — nine feeds, one brain; GAVI flags the anomaly (a fall).
  B. Site view — "this is what's happening": workers tracked, PPE rule fires.
  C. Aerial view — GAVI points at things from drone footage.

The acts are composed with cv2 + ffmpeg (h264 for browsers), then each act is
run through the REAL pipeline (fresh world per act, its own watch rule) and
the boxes/events/rule-verdicts are saved to web/assets/hero_tracks.json for
the landing page to replay in sync. Nothing in the overlay is invented.

Usage:  python make_hero.py            (expects the clips in web/assets/)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from run import load_dotenv

load_dotenv()
from backends import make_backend
from engine import run_cycle
from world import WorldModel

ASSETS = Path(__file__).parent / "web" / "assets"
W, H, FPS = 1280, 720, 24

# (clip file, in-point seconds, seconds to use)
ACT_A_SECS = 7.0  # grid wall
ACT_B = ("demo_site.mp4", 1.0, 5.0)
ACT_C = ("demo_aerial.mp4", 1.0, 5.0)

GRID_CELLS = [  # row-major: (clip, label)
    ("hero_plaza.mp4", "CAM 01"),
    ("demo_twins.mp4", "CAM 02"),
    ("demo_loitering.mp4", "CAM 03"),
    ("demo_permanence.mp4", "CAM 04"),
    ("demo_fall.mp4", "CAM 05"),  # the anomaly, center cell
    ("demo_site.mp4", "CAM 06"),
    ("demo_aerial.mp4", "CAM 07"),
    ("demo_twins.mp4", "CAM 08"),
    ("demo_loitering.mp4", "CAM 09"),
]
ACT_A_INPOINT = {"demo_fall.mp4": 1.5}  # start the fall clip where it matters

RULES = {
    "A": "someone has fallen or is lying on the ground",
    "B": "a worker is not wearing a hard hat",
    "C": "a truck is parked blocking the gate",
}
LABELS = {
    "A": "ACT 1 · NINE FEEDS, ONE BRAIN",
    "B": "ACT 2 · IT KNOWS WHAT'S HAPPENING",
    "C": "ACT 3 · ANY VANTAGE POINT",
}


def read_clip(path: Path, start: float, secs: float, size) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    frames = []
    for _ in range(int(secs * fps)):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size))
    cap.release()
    # resample to our FPS by index if the clip fps differs
    if abs(fps - FPS) > 0.5 and frames:
        idx = np.linspace(0, len(frames) - 1, int(secs * FPS)).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def cell_chrome(f: np.ndarray, label: str, t: float) -> np.ndarray:
    """CCTV dressing: label bar + timecode."""
    f = f.copy()
    cv2.rectangle(f, (0, 0), (f.shape[1], 20), (12, 12, 14), -1)
    cv2.putText(f, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 210), 1)
    tc = time.strftime("%H:%M:", time.gmtime(0)) + f"{int(t)%60:02d}"
    cv2.putText(f, "REC  13:0" + f"{int(t)%10}" + f":{int(t*24)%60:02d}",
                (f.shape[1] - 118, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 140), 1)
    return f


def build_video() -> list[tuple[str, float, float]]:
    """Compose hero.mp4; return [(act, start, end)] in final-video seconds."""
    cw, ch = W // 3, H // 3
    acts, frames = [], []

    # --- Act A: the wall ---
    cells = []
    for clip, label in GRID_CELLS:
        start = ACT_A_INPOINT.get(clip, 0.5)
        cells.append((read_clip(ASSETS / clip, start, ACT_A_SECS, (cw, ch)), label))
    n = min(len(c[0]) for c in cells)
    a0 = 0.0
    for i in range(n):
        rows = []
        for r in range(3):
            row = [cell_chrome(cells[r * 3 + c][0][i], cells[r * 3 + c][1], i / FPS)
                   for c in range(3)]
            rows.append(np.hstack(row))
        mosaic = np.vstack(rows)
        # 3*(W//3) != W — resize to the exact output size or the writer
        # silently drops every mosaic frame.
        frames.append(cv2.resize(mosaic, (W, H)))
    acts.append(("A", a0, len(frames) / FPS))

    # --- Acts B & C: full-frame views ---
    for act, (clip, start, secs) in (("B", ACT_B), ("C", ACT_C)):
        s0 = len(frames) / FPS
        frames.extend(read_clip(ASSETS / clip, start, secs, (W, H)))
        acts.append((act, s0, len(frames) / FPS))

    # write raw, then transcode to h264 (browsers won't play cv2's mp4v)
    tmp = tempfile.mktemp(suffix=".mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
         "-movflags", "+faststart", str(ASSETS / "hero.mp4")],
        check=True,
    )
    Path(tmp).unlink()
    print(f"hero.mp4 written: {len(frames)/FPS:.1f}s, acts: {acts}")
    return acts


def trace_act(video: Path, backend, act: str, start: float, end: float) -> dict:
    """Run the real pipeline over one act; return tracks/events/alerts."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    wdir = tempfile.mkdtemp(prefix=f"hero-act-{act}-")
    world = WorldModel(wdir, resume=False)
    tracks, events, alerts = [], [], []
    buf = []
    i0, i1 = int(start * fps), int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    for i in range(i0, i1):
        ok, frame = cap.read()
        if not ok:
            break
        if (i - i0) % 8 == 0:  # ~3 samples/sec
            vt = i / fps
            buf.append(frame)
            res = run_cycle(world, backend, buf[-3:], span=min(len(buf), 3) * 8 / fps,
                            rule=RULES[act])
            if not res.ok:
                continue
            tracks.append({"t": round(vt, 2), "boxes": res.boxes})
            for ev in res.events:
                events.append({"t": round(vt, 2), "type": ev.type, "text": ev.text})
            v = res.rule_verdict
            if v and v.triggered and v.confidence >= 0.6:
                alerts.append({"t": round(vt, 2), "text": v.reason})
    cap.release()
    shutil.rmtree(wdir, ignore_errors=True)
    print(f"act {act}: {len(tracks)} samples, {len(events)} events, {len(alerts)} rule hits")
    return {"act": act, "start": round(start, 2), "end": round(end, 2),
            "label": LABELS[act], "rule": RULES[act],
            "tracks": tracks, "events": events, "alerts": alerts}


def main() -> None:
    acts = build_video()
    backend = make_backend("gemini", "gemini-3.1-flash-lite", "balanced")
    segments = [trace_act(ASSETS / "hero.mp4", backend, a, s, e) for a, s, e in acts]
    out = ASSETS / "hero_tracks.json"
    json.dump({"version": 2, "segments": segments}, open(out, "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
