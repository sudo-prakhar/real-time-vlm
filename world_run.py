#!/usr/bin/env python3
"""Long-horizon world-model monitor (Phase 2).

Where run.py judges each frame in isolation, this keeps a persistent world
model: a registry of entities (tracked across exits/re-entries — object
permanence) and an event timeline including *inferences* about what happened
off-screen. Each cycle sends the VLM the current frame PLUS the world state;
the model returns a delta that world.WorldModel folds in and persists.

Concurrency model: unlike run.py's stateless worker pool, updates are serial —
each call must see the state the previous call produced — so there is ONE
updater thread. Q&A runs on its own thread so questions never stall tracking.

Live control (type in the terminal):
    ? where is the red mug          -> ask the world model a question
    the person has been gone > 1m   -> replace the watch rule (blank line clears)

Examples:
    python world_run.py --backend gemini --model gemini-3.1-flash-lite --display
    python world_run.py --rule "the desk is left unattended" --fresh
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime

import cv2

from backends import Verdict, make_backend
from engine import MotionGate, parse_box, run_cycle  # parse_box re-exported for tests/tools
from run import c, load_dotenv, notify, open_source
from world import WorldModel

EVENT_COLORS = {
    "entered": "1;92",
    "exited": "1;93",
    "inference": "1;95",
    "behavior": "1;94",
    "identity_uncertain": "33",
}


class WorldMonitor:
    """Shared state between the capture loop, updater, and Q&A threads."""

    BUFFER_SPACING = 0.15  # min seconds between buffered frames
    BUFFER_HORIZON = 10.0  # how far back the burst buffer reaches

    def __init__(self, args, world: WorldModel, backend) -> None:
        self.args = args
        self.world = world
        self.backend = backend
        self.gate = (
            MotionGate(args.motion_threshold, args.heartbeat)
            if args.motion_gate
            else None
        )
        self.rule = args.rule
        self._lock = threading.Lock()
        self._buffer: list[tuple[float, object]] = []  # (t, frame), oldest first
        self.status = "warming up…"
        self.triggered = False
        self._hits = 0
        self._last_alert = float("-inf")

    def submit(self, frame) -> None:
        now = time.monotonic()
        with self._lock:
            if self._buffer and now - self._buffer[-1][0] < self.BUFFER_SPACING:
                self._buffer[-1] = (self._buffer[-1][0], frame)  # keep newest crisp
                return
            self._buffer.append((now, frame))
            while self._buffer and now - self._buffer[0][0] > self.BUFFER_HORIZON:
                self._buffer.pop(0)

    def latest(self):
        with self._lock:
            return None if not self._buffer else self._buffer[-1][1].copy()

    def burst(self, k: int, span: float) -> tuple[list, float]:
        """Up to k buffered frames evenly spread over the last `span` seconds,
        oldest first, always ending on the newest frame. Returns (frames, actual span)."""
        with self._lock:
            if not self._buffer:
                return [], 0.0
            now = self._buffer[-1][0]
            window = [(t, f) for t, f in self._buffer if now - t <= span]
            if k <= 1 or len(window) <= 1:
                return [self._buffer[-1][1].copy()], 0.0
            idx = sorted({round(i * (len(window) - 1) / (k - 1)) for i in range(k)})
            picked = [window[i] for i in idx]
            return [f.copy() for _, f in picked], picked[-1][0] - picked[0][0]

    def set_rule(self, rule: str) -> None:
        with self._lock:
            self.rule = rule or None
            self._hits = 0

    def handle_rule_verdict(self, verdict: Verdict, frame) -> None:
        with self._lock:
            self.triggered = verdict.triggered
            self._hits = self._hits + 1 if verdict.triggered else 0
            now = time.monotonic()
            fire = (
                self._hits >= self.args.consecutive
                and (now - self._last_alert) >= self.args.cooldown
            )
            rule = self.rule
            if fire:
                self._last_alert = now
                self._hits = 0
        if verdict.triggered:
            print(c(f"  rule: {verdict.confidence:.0%} {verdict.reason}", "1;91"), flush=True)
        if fire and rule:
            notify(rule, verdict, frame, self.args.evidence_dir)


def update_loop(mon: WorldMonitor, stop: threading.Event) -> None:
    """The serial world-update cycle: OBSERVE (vision, memory-blind) ->
    MATCH (code: appearance + motion continuity) -> REASON (text, memory-aware,
    skipped when nothing changed). See world.py for why it's split this way."""
    args, world = mon.args, mon.world
    thumb_dir = os.path.join(args.world_dir, "entities")
    while not stop.is_set():
        try:
            _update_once(mon, stop, thumb_dir)
        except Exception:  # a monitor must outlive any single bad cycle
            import traceback

            print(c(f"  update cycle failed:\n{traceback.format_exc()}", "33"), flush=True)
            time.sleep(1.0)


def _update_once(mon: WorldMonitor, stop: threading.Event, thumb_dir: str) -> None:
    """One observe -> match -> apply -> reason cycle (engine.run_cycle) plus
    the terminal-specific parts: logging, rule debounce, pacing."""
    args, world = mon.args, mon.world
    frames, span = mon.burst(args.burst, args.burst_span)
    if not frames:
        time.sleep(0.05)
        return
    # Static scene -> no VLM spend. A tighter heartbeat while a rule is active
    # keeps temporal rules ("gone for a minute") advancing between motions.
    if mon.gate and not mon.gate.should_process(
        frames[-1], heartbeat=args.rule_heartbeat if mon.rule else None
    ):
        time.sleep(0.15)
        return
    try:
        res = run_cycle(
            world, mon.backend, frames, span,
            rule=mon.rule, max_side=args.max_side, thumb_dir=thumb_dir,
            # A gated gap is verifiably static, so presence spans it fully.
            presence_credit=args.heartbeat + 5.0 if mon.gate else 5.0,
        )
    except Exception as e:  # keep tracking alive on transient backend errors
        mon.status = "backend error"
        print(c(f"  backend error: {e}", "33"), flush=True)
        time.sleep(0.5)
        return
    if not res.ok:
        print(c(f"  {res.error} — skipping frame", "33"), flush=True)
        return

    if mon.rule and res.rule_verdict:
        mon.handle_rule_verdict(res.rule_verdict, frames[-1])

    mon.status = f"{res.present} in frame · {res.total} tracked · {res.latency:.2f}s"
    print(
        c(f"  {datetime.now():%H:%M:%S}  [{mon.status}]  {res.scene}", "90"),
        flush=True,
    )
    for ev in res.events:
        conf = f" ({ev.confidence:.0%})" if ev.type in ("inference", "behavior") else ""
        print(c(f"    ∙ {ev.type}: {ev.text}{conf}", EVENT_COLORS.get(ev.type, "37")), flush=True)

    if args.interval:
        stop.wait(args.interval)


def stdin_loop(mon: WorldMonitor, stop: threading.Event) -> None:
    """`? question` asks the world model; anything else replaces the rule."""
    for line in sys.stdin:
        if stop.is_set():
            break
        line = line.strip()
        if line.startswith("?"):
            question = line.lstrip("? ").strip()
            if not question:
                continue
            prompt = mon.world.build_answer_prompt(question)
            try:
                answer = mon.backend.generate(prompt, json_mode=False)
            except Exception as e:
                print(c(f"  answer error: {e}", "33"), flush=True)
                continue
            print(c(f"  ❓ {question}", "96"), flush=True)
            print(c(f"  💬 {answer.strip()}", "1;96"), flush=True)
        else:
            mon.set_rule(line)
            msg = f"rule updated: {line!r}" if line else "rule cleared"
            print(c(f"  → {msg}", "1;96"), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Long-horizon world-model monitor (Phase 2)")
    p.add_argument("--rule", default=None, help="Optional watch rule — may be temporal ('the mug has been gone for 5 minutes')")
    p.add_argument("--source", default="0", help="Webcam index, file path, or RTSP/HTTP URL (default: 0)")
    p.add_argument("--backend", default="ollama", choices=["ollama", "gemini"])
    p.add_argument("--model", default=None, help="Override the backend's default model")
    p.add_argument("--interval", type=float, default=0.5, help="Min seconds between world updates (default 0.5; serial anyway)")
    p.add_argument("--burst", type=int, default=3, help="Frames per observe call (default 3). >1 lets the model see motion — key for telling identical-looking entities apart. 1 = single frame.")
    p.add_argument("--burst-span", type=float, default=2.0, help="Seconds the burst reaches back (default 2.0)")
    p.add_argument("--max-side", type=int, default=768, help="Downscale frames before the VLM (default 768; 0 = full res)")
    p.add_argument("--no-motion-gate", dest="motion_gate", action="store_false", help="Run cycles even when the scene is static (default: skip them — motion still triggers a cycle instantly)")
    p.add_argument("--motion-threshold", type=float, default=0.01, help="Changed-pixel fraction vs the last processed frame that counts as motion (default 0.01)")
    p.add_argument("--heartbeat", type=float, default=60.0, help="Max seconds between cycles on a static scene (default 60)")
    p.add_argument("--rule-heartbeat", type=float, default=15.0, help="Heartbeat while a rule is active, so temporal rules keep advancing (default 15)")
    p.add_argument("--world-dir", default="world", help="Where state.json + timeline.jsonl live")
    p.add_argument("--fresh", action="store_true", help="Start a new world instead of resuming from --world-dir")
    p.add_argument("--consecutive", type=int, default=2, help="Positive rule checks before alerting (default 2)")
    p.add_argument("--cooldown", type=float, default=30.0, help="Min seconds between repeat alerts (default 30)")
    p.add_argument("--display", action="store_true", help="Show the video with world-state overlay (press q to quit)")
    p.add_argument("--evidence-dir", default="evidence")
    args = p.parse_args()

    load_dotenv()
    backend = make_backend(args.backend, args.model, sensitivity="balanced")
    if args.fresh:
        for name in ("state.json", "timeline.jsonl"):
            path = os.path.join(args.world_dir, name)
            if os.path.exists(path):
                os.remove(path)
    world = WorldModel(args.world_dir, resume=not args.fresh)

    cap = open_source(args.source)
    _, total = world.counts()
    resumed = f", resuming a world with {total} known entities" if total else ""
    print(f"World-model monitor [{args.backend}]{resumed}", flush=True)
    if args.rule:
        print(f"Watching rule: {args.rule!r}", flush=True)
    print(c("Type `? <question>` to query the world · a rule + Enter to watch it · Ctrl+C to stop.", "96"), flush=True)

    mon = WorldMonitor(args, world, backend)
    stop = threading.Event()
    threading.Thread(target=update_loop, args=(mon, stop), daemon=True).start()
    threading.Thread(target=stdin_loop, args=(mon, stop), daemon=True).start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file or camera dropped
            mon.submit(frame)
            if args.display:
                view = frame.copy()
                fh, fw = view.shape[:2]
                for eid, b in world.present_boxes():
                    p0, p1 = (int(b[0] * fw), int(b[1] * fh)), (int(b[2] * fw), int(b[3] * fh))
                    cv2.rectangle(view, p0, p1, (0, 200, 0), 2)
                    cv2.putText(view, eid, (p0[0], max(14, p0[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                color = (0, 0, 255) if mon.triggered else (0, 200, 0)
                cv2.putText(view, mon.status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(view, world.scene[:80], (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                if mon.rule:
                    cv2.putText(view, f"rule: {mon.rule[:64]}", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                cv2.imshow("world monitor", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        world.save()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        present, total = world.counts()
        print(f"\nWorld saved to {args.world_dir}/ — {total} entities, {len(world.events)} events.", flush=True)
        if hasattr(backend, "usage_line"):
            gate_note = f"  ({mon.gate.skipped} static cycles skipped)" if mon.gate else ""
            print(f"API usage: {backend.usage_line()}{gate_note}", flush=True)


if __name__ == "__main__":
    main()
