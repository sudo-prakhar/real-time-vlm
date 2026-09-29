"""Real-time VLM monitor (POC).

Phase-1 pure-VLM cascade: sample frames from a video source, ask the VLM whether
a plain-English rule is true, debounce, and notify. The backend is pluggable
(local Ollama or cloud Gemini).

Concurrency model:
  - The capture/display loop runs on the main thread (never blocks on inference).
  - A POOL of N worker threads each grab the latest frame and call the VLM. Cloud
    calls are I/O-bound, so N in-flight calls raise throughput ~linearly:
    effective check rate ≈ N / per_call_latency.  Set with --workers.

Live control:
  - Each log line shows the capture→response latency.
  - Type a new rule + Enter in the terminal to change what's being watched, live.

Examples:
    python -m gavi monitor --backend gemini --model gemini-3.1-flash-lite --workers 6 \\
        --interval 0 --rule "a person is not wearing a hard hat" --display
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime

import cv2

from gavi.backends import Verdict, make_backend
from gavi.utils import c, load_dotenv, notify
from gavi.video import downscale, motion_fraction, open_source


class Monitor:
    """Shared state for a pool of concurrent inference workers."""

    def __init__(self, args) -> None:
        self.args = args
        self.rule = args.rule  # live — editable via the stdin listener
        self._lock = threading.Lock()
        self._frame = None
        self._frame_t = 0.0
        self.status = "warming up…"  # read by the display loop for the overlay
        self.triggered = False
        self._hits = 0
        self._last_alert = float("-inf")

    def set_rule(self, rule: str) -> None:
        with self._lock:
            self.rule = rule
            self._hits = 0  # don't carry a stale hit count across rule changes

    def submit(self, frame) -> None:
        with self._lock:
            self._frame = frame
            self._frame_t = time.monotonic()  # capture timestamp for latency

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_t

    def record(self, verdict: Verdict, frame, latency: float) -> None:
        """Fold one worker's verdict into the debounce state; alert if tripped."""
        with self._lock:
            self.triggered = verdict.triggered
            self.status = (
                f"{'TRIGGERED' if verdict.triggered else 'ok'} "
                f"{verdict.confidence:.0%} · {latency:.2f}s"
            )
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
        line = f"  {datetime.now():%H:%M:%S}  {self.status}  {verdict.reason}"
        # Grey when merely noticing; bold bright-red when a frame trips the rule.
        print(c(line, "1;91" if verdict.triggered else "90"), flush=True)
        if fire:
            notify(rule, verdict, frame, self.args.evidence_dir)


def worker_loop(monitor: Monitor, backend, args, stop: threading.Event, idx: int) -> None:
    # Stagger startup so N workers desync and spread their calls over time.
    time.sleep((args.interval or 0.3) * idx / max(args.workers, 1))
    prev_gray = None
    while not stop.is_set():
        if args.interval:
            time.sleep(args.interval)
        frame, _ = monitor.latest()
        if frame is None:
            time.sleep(0.05)
            continue

        if args.motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            static = (
                prev_gray is not None
                and motion_fraction(prev_gray, gray) < args.motion_threshold
            )
            prev_gray = gray
            if static:
                continue

        t0 = time.monotonic()
        ok, buf = cv2.imencode(".jpg", downscale(frame, args.max_side))
        if not ok:
            continue
        try:
            verdict = backend.check(buf.tobytes(), monitor.rule)
        except Exception as e:  # keep the pool alive on transient backend errors
            monitor.status = "backend error"
            print(c(f"  backend error: {e}", "33"), flush=True)  # yellow
            time.sleep(0.5)
            continue
        latency = time.monotonic() - t0  # encode + inference for this frame
        monitor.record(verdict, frame, latency)


def stdin_rule_listener(monitor: Monitor, stop: threading.Event) -> None:
    """Type a new rule + Enter in the terminal to change it live."""
    for line in sys.stdin:
        if stop.is_set():
            break
        rule = line.strip()
        if rule:
            monitor.set_rule(rule)
            print(c(f"  → rule updated: {rule!r}", "1;96"), flush=True)  # bold cyan


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m gavi monitor", description="Real-time VLM monitor (POC)")
    p.add_argument("--rule", required=True, help="Plain-English condition to watch for (editable live via stdin)")
    p.add_argument("--source", default="0", help="Webcam index, file path, or RTSP/HTTP URL (default: 0)")
    p.add_argument("--backend", default="ollama", choices=["ollama", "gemini"])
    p.add_argument("--model", default=None, help="Override the backend's default model")
    p.add_argument("--sensitivity", default=None, choices=["liberal", "balanced", "strict"], help="How eagerly to trigger (default: ollama=liberal/high-recall, gemini=strict)")
    p.add_argument("--workers", type=int, default=1, help="Concurrent in-flight VLM calls (default 1). Cloud is I/O-bound, so 3-6 raises throughput ~linearly until rate limits.")
    p.add_argument("--interval", type=float, default=1.0, help="Min seconds between a worker's checks (default 1.0; use 0 for back-to-back)")
    p.add_argument("--max-side", type=int, default=0, help="Downscale frames so the longest side <= N px before the VLM (big local speedup; try 512). 0 = full res.")
    p.add_argument("--consecutive", type=int, default=2, help="Positive checks before alerting (default 2)")
    p.add_argument("--cooldown", type=float, default=30.0, help="Min seconds between repeat alerts (default 30)")
    p.add_argument("--motion", action="store_true", help="Skip checks on near-static frames (cheap motion gate)")
    p.add_argument("--motion-threshold", type=float, default=0.01, help="Changed-pixel fraction that counts as motion")
    p.add_argument("--display", action="store_true", help="Show the video window with a live status overlay (press q to quit)")
    p.add_argument("--evidence-dir", default="evidence")
    args = p.parse_args(argv)

    load_dotenv()  # pick up GEMINI_API_KEY from .env if present
    backend = make_backend(args.backend, args.model, args.sensitivity)
    cap = open_source(args.source)
    print(
        f"Monitoring [{args.backend}] for rule: {args.rule!r}  "
        f"({args.workers} worker{'s' if args.workers != 1 else ''})",
        flush=True,
    )
    print(c("Type a new rule + Enter to change it live.  Ctrl+C to stop.", "96"), flush=True)

    monitor = Monitor(args)
    stop = threading.Event()
    workers = [
        threading.Thread(target=worker_loop, args=(monitor, backend, args, stop, i), daemon=True)
        for i in range(args.workers)
    ]
    for w in workers:
        w.start()
    threading.Thread(target=stdin_rule_listener, args=(monitor, stop), daemon=True).start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file or camera dropped
            monitor.submit(frame)

            if args.display:
                view = frame.copy()
                color = (0, 0, 255) if monitor.triggered else (0, 200, 0)
                cv2.putText(view, monitor.status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(view, f"rule: {monitor.rule[:64]}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                cv2.imshow("monitor", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.005)  # don't busy-spin when headless
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
