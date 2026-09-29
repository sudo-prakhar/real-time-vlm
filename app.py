#!/usr/bin/env python3
"""GAVI web app: the landing page + live browser demos.

Two demo modes share one pipeline (engine.run_cycle):
  - Preset demos: the server streams a bundled clip through the world model
    and pushes both the video frames and the model's understanding to the
    browser — visitors watch GAVI watch the video.
  - Live demo: the browser streams webcam frames up instead.

Each WebSocket session owns one isolated, ephemeral world (temp dir, deleted
on disconnect). `stop` halts feeding + processing without dropping the world,
so the timeline stays on screen.

Run:  make web   →   http://localhost:8000
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backends import make_backend
from engine import MotionGate, run_cycle
from run import load_dotenv
from world import WorldModel

load_dotenv()

WEB_DIR = Path(__file__).parent / "web"
ASSETS = WEB_DIR / "assets"
BACKEND_NAME = "gemini"
BACKEND_MODEL = "gemini-3.1-flash-lite"
SESSION_MAX_SECS = 15 * 60
MAX_SESSIONS = 6  # each live session costs Gemini calls — cap the public tap
_active_sessions = 0
BUFFER_HORIZON = 10.0
BUFFER_SPACING = 0.15
BURST, BURST_SPAN = 3, 2.0
MOTION_THRESHOLD = 0.01  # changed-pixel fraction vs the last processed frame
HEARTBEAT = 60.0  # max seconds between cycles on a static scene
RULE_HEARTBEAT = 6.0  # tighter while a rule is set — "held X for >10s"-style rules need cycles near the crossing
PRESET_LOOPS = 1  # play each demo clip through exactly once
PRESETS = {
    "permanence": ASSETS / "demo_permanence.mp4",
    "twins": ASSETS / "demo_twins.mp4",
    "loitering": ASSETS / "demo_loitering.mp4",
    "safety": ASSETS / "demo_fall.mp4",
    "site": ASSETS / "demo_site.mp4",
    "aerial": ASSETS / "demo_aerial.mp4",
}

app = FastAPI(title="GAVI")
if ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
_backend = None


def backend():
    global _backend
    if _backend is None:  # lazy: loading the page shouldn't require an API key
        _backend = make_backend(BACKEND_NAME, BACKEND_MODEL, sensitivity="balanced")
    return _backend


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/about")
async def about():
    return FileResponse(WEB_DIR / "about.html")


class Session:
    """One visitor: a frame buffer + an isolated world + at most one feeder."""

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="gavi-session-")
        self.world = WorldModel(self.dir, resume=False)
        self.frames: list[tuple[float, np.ndarray]] = []
        self.rule: str | None = None
        self.feeder: asyncio.Task | None = None
        self.gate = MotionGate(MOTION_THRESHOLD, HEARTBEAT)
        self.last_processed = 0.0  # newest frame ts the cycle loop has consumed
        self.processing = False  # a cycle is in flight right now

    def drained(self) -> bool:
        """True when every buffered frame has been consumed and no cycle is
        mid-flight — i.e. 'clip finished' would genuinely be the last word."""
        return not self.processing and not (
            self.frames and self.frames[-1][0] > self.last_processed
        )

    def reset_world(self) -> None:
        """Fresh world for a fresh demo run; the old temp dir is discarded."""
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir = tempfile.mkdtemp(prefix="gavi-session-")
        self.world = WorldModel(self.dir, resume=False)
        self.frames = []
        self.rule = None
        self.gate.reset()  # a fresh demo must get an immediate first cycle
        self.last_processed = 0.0

    def stop_feeding(self) -> None:
        if self.feeder:
            self.feeder.cancel()
            self.feeder = None
        self.frames = []  # nothing to process -> the cycle loop idles

    def add_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if self.frames and now - self.frames[-1][0] < BUFFER_SPACING:
            self.frames[-1] = (self.frames[-1][0], frame)
            return
        self.frames.append((now, frame))
        while self.frames and now - self.frames[0][0] > BUFFER_HORIZON:
            self.frames.pop(0)

    def burst(self, k: int, span: float) -> tuple[list, float, float]:
        """(frames oldest-first, span, newest timestamp) — timestamp lets the
        cycle loop skip when no new frame arrived since the last cycle."""
        if not self.frames:
            return [], 0.0, 0.0
        now = self.frames[-1][0]
        window = [(t, f) for t, f in self.frames if now - t <= span]
        if k <= 1 or len(window) <= 1:
            return [self.frames[-1][1]], 0.0, now
        idx = sorted({round(i * (len(window) - 1) / (k - 1)) for i in range(k)})
        picked = [window[i] for i in idx]
        return [f for _, f in picked], picked[-1][0] - picked[0][0], now

    def close(self) -> None:
        self.stop_feeding()
        shutil.rmtree(self.dir, ignore_errors=True)


async def _feed_preset(session: Session, outbox: asyncio.Queue, name: str) -> None:
    """Stream a preset clip: frames into the session buffer for the world
    model, and downsized JPEGs to the browser for display."""
    path = PRESETS[name]
    for _ in range(PRESET_LOOPS):
        # A fresh capture per loop — seeking back to frame 0 on an open
        # capture silently fails for some codecs, ending the loop instantly.
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        step = max(1, round(fps / 8))  # ~8 frames/sec is plenty for both consumers
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    session.add_frame(frame)
                    h, w = frame.shape[:2]
                    small = cv2.resize(frame, (560, int(560 * h / w)))
                    ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 62])
                    if ok2:
                        await outbox.put({
                            "type": "video_frame",
                            "jpeg": base64.b64encode(buf.tobytes()).decode(),
                        })
                    await asyncio.sleep(step / fps)  # real-time pacing
                i += 1
        finally:
            cap.release()
    # Let the pipeline drain before announcing the end — otherwise the last
    # cycle's understanding lands AFTER "clip finished" in the feed.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not session.drained():
        await asyncio.sleep(0.15)
    await outbox.put({"type": "preset_ended", "name": name})


async def _cycle_loop(session: Session, outbox: asyncio.Queue) -> None:
    loop = asyncio.get_event_loop()
    started = time.monotonic()
    while True:
        if time.monotonic() - started > SESSION_MAX_SECS:
            await outbox.put({"type": "expired"})
            return
        frames, span, newest = session.burst(BURST, BURST_SPAN)
        if not frames or newest <= session.last_processed:
            await asyncio.sleep(0.1)  # stopped or no fresh frame -> spend nothing
            continue
        if not session.gate.should_process(
            frames[-1], heartbeat=RULE_HEARTBEAT if session.rule else None
        ):
            session.last_processed = newest  # static scene -> spend nothing
            await asyncio.sleep(0.1)
            continue
        session.last_processed = newest
        world = session.world  # bind: reset_world() may swap mid-cycle
        session.processing = True  # cleared only after the update is queued
        try:
            try:
                res = await loop.run_in_executor(
                    None,
                    lambda: run_cycle(
                        world, backend(), frames, span, rule=session.rule,
                        # A gated gap is verifiably static -> presence spans it.
                        presence_credit=HEARTBEAT + 5.0,
                    ),
                )
            except Exception as e:
                await outbox.put({"type": "error", "text": str(e)[:200]})
                await asyncio.sleep(1.0)
                continue
            if not res.ok or world is not session.world:
                continue  # glitch, or a new demo started while we were thinking
            present_ids = {b["id"] for b in res.boxes}
            await _put_update(outbox, session, world, res, present_ids)
        finally:
            session.processing = False


async def _put_update(outbox, session, world, res, present_ids) -> None:
    await outbox.put({
            "type": "update",
            "scene": res.scene,
            "boxes": res.boxes,
            "events": [asdict(ev) for ev in res.events],
            "entities": [
                {
                    "id": ent.id,
                    "label": ent.label,
                    "description": ent.description,
                    "present": ent.id in present_ids,
                    "dwell": round(ent.total_visible, 1),
                }
                for ent in world.tracked()
            ],
            "present": res.present,
            "total": res.total,
            "latency": round(res.latency, 2),
            "thumbs": {k: base64.b64encode(v).decode() for k, v in res.new_thumbs.items()},
            "rule": (
                {
                    "text": session.rule,
                    "triggered": res.rule_verdict.triggered,
                    "confidence": res.rule_verdict.confidence,
                    "reason": res.rule_verdict.reason,
                }
                if session.rule and res.rule_verdict
                else None
            ),
        })


async def _sender(ws: WebSocket, outbox: asyncio.Queue) -> None:
    """Single writer — Starlette websockets don't tolerate concurrent sends."""
    while True:
        await ws.send_json(await outbox.get())


@app.websocket("/ws")
async def demo_socket(ws: WebSocket) -> None:
    global _active_sessions
    await ws.accept()
    if _active_sessions >= MAX_SESSIONS:
        await ws.send_json({
            "type": "error",
            "text": "GAVI is at capacity right now — try again in a few minutes.",
        })
        await ws.close()
        return
    _active_sessions += 1
    session = Session()
    usage_at_start = dict(_backend.usage) if _backend else None
    outbox: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.ensure_future(_cycle_loop(session, outbox)),
        asyncio.ensure_future(_sender(ws, outbox)),
    ]
    loop = asyncio.get_event_loop()
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "frame":  # live webcam frame from the browser
                raw = base64.b64decode(msg.get("jpeg", ""))
                arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    session.add_frame(arr)
            elif kind == "start_preset" and msg.get("name") in PRESETS:
                session.stop_feeding()
                session.reset_world()
                session.feeder = asyncio.ensure_future(
                    _feed_preset(session, outbox, msg["name"])
                )
                await outbox.put({"type": "started", "mode": "preset", "name": msg["name"]})
            elif kind == "start_live":
                session.stop_feeding()
                session.reset_world()
                await outbox.put({"type": "started", "mode": "live"})
            elif kind == "stop":
                session.stop_feeding()
                await outbox.put({"type": "stopped"})
            elif kind == "ask":
                question = str(msg.get("text", "")).strip()
                if question:
                    prompt = session.world.build_answer_prompt(question)
                    answer = await loop.run_in_executor(
                        None, lambda: backend().generate(prompt, json_mode=False)
                    )
                    await outbox.put(
                        {"type": "answer", "question": question, "text": answer.strip()}
                    )
            elif kind == "rule":
                session.rule = str(msg.get("text", "")).strip() or None
                await outbox.put({"type": "rule_set", "text": session.rule})
    except WebSocketDisconnect:
        pass
    finally:
        _active_sessions -= 1
        for t in tasks:
            t.cancel()
        session.close()
        if _backend is not None:
            # Delta since this session started — approximate when sessions
            # overlap (the backend is shared), exact when they don't.
            base = usage_at_start or {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            delta = {k: _backend.usage[k] - base.get(k, 0) for k in _backend.usage}
            print(
                f"session closed: {_backend.usage_line(delta)}"
                f"  ({session.gate.skipped} static cycles skipped)",
                flush=True,
            )
