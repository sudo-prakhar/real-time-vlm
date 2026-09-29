"""The shared world-update cycle: OBSERVE -> MATCH -> APPLY -> REASON.

Both runtimes drive this one function — world_monitor.py (terminal) and
server.py (web) — so the pipeline can't drift between them. It takes a burst
of frames and a WorldModel, runs one full perception cycle, and returns
everything a frontend needs to render: events, present boxes, new thumbnails,
rule verdict.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2

from gavi import identity
from gavi.backends import Verdict, extract_json
from gavi.video import downscale, motion_fraction
from gavi.world import Event, WorldModel


class MotionGate:
    """Skip VLM cycles while the scene is static — the biggest cost lever, and
    latency-free: motion trips the gate on the very next poll, so an event is
    processed exactly as fast as without the gate.

    The diff is against the last PROCESSED frame (not the previous poll), so
    slow drift accumulates until it crosses the threshold instead of hiding
    below it frame-to-frame. A heartbeat forces a cycle every `heartbeat`
    seconds regardless, so time-based reasoning ("gone for a minute", dwell
    stats) still advances on a static scene — pass a tighter per-call
    `heartbeat` while a temporal rule is active.
    """

    SIZE = (160, 90)  # diff resolution: plenty to catch a person, ~50µs to compute

    def __init__(self, threshold: float = 0.01, heartbeat: float = 60.0) -> None:
        self.threshold = threshold
        self.heartbeat = heartbeat
        self.skipped = 0  # cycles not sent to the VLM (money not spent)
        self._gray = None
        self._t = 0.0

    def reset(self) -> None:
        self._gray, self._t, self.skipped = None, 0.0, 0

    def should_process(self, frame, heartbeat: float | None = None) -> bool:
        now = time.monotonic()
        hb = self.heartbeat if heartbeat is None else heartbeat
        gray = cv2.cvtColor(cv2.resize(frame, self.SIZE), cv2.COLOR_BGR2GRAY)
        if (
            self._gray is None
            or now - self._t >= hb
            or motion_fraction(self._gray, gray) >= self.threshold
        ):
            self._gray, self._t = gray, now
            return True
        self.skipped += 1
        return False


def parse_box(raw, box_format: str = "xyxy") -> list[float] | None:
    """Coerce a model-reported box to normalized [x0,y0,x1,y1] in 0..1."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if box_format == "yxyx":
        vals = [vals[1], vals[0], vals[3], vals[2]]
    # Prompt asks for 0-1000; some models emit 0-1 already. A "0-1000" box
    # with every coord <= 1.5 would be sub-pixel garbage, so treat as 0-1.
    scale = 1.0 if max(vals) <= 1.5 else 1000.0
    x0, y0, x1, y1 = (min(max(v / scale, 0.0), 1.0) for v in vals)
    if x1 - x0 < 0.005 or y1 - y0 < 0.005:
        return None
    return [x0, y0, x1, y1]


@dataclass
class CycleResult:
    ok: bool
    error: str = ""
    scene: str = ""
    events: list[Event] = field(default_factory=list)
    rule_verdict: Verdict | None = None
    boxes: list[dict] = field(default_factory=list)  # {id, label, box} visible now
    new_thumbs: dict = field(default_factory=dict)  # entity id -> jpeg bytes (first sighting)
    present: int = 0
    total: int = 0
    latency: float = 0.0


def run_cycle(
    world: WorldModel,
    backend,
    frames: list,
    span: float,
    rule: str | None = None,
    max_side: int = 768,
    thumb_dir: str | None = None,
    presence_credit: float = 5.0,
) -> CycleResult:
    """One full cycle over a burst of frames (oldest first, newest last)."""
    t0 = time.monotonic()
    frame = frames[-1]  # boxes, signatures, and thumbnails anchor to the newest
    jpegs = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", downscale(f, max_side))
        if ok:
            jpegs.append(buf.tobytes())
    if not jpegs:
        return CycleResult(ok=False, error="could not encode frames")

    now = time.time()
    box_format = getattr(backend, "box_format", "xyxy")
    observation = extract_json(
        backend.generate(
            world.build_observe_prompt(box_format, burst=len(jpegs), span=span),
            images=jpegs,
        )
    )
    if observation is None:
        return CycleResult(ok=False, error="unparseable observation")

    # Geometry + appearance signatures (computed on the full-res frame).
    obs_entities = [e for e in observation.get("entities", []) or [] if isinstance(e, dict)]
    observation["entities"] = obs_entities
    h, w = frame.shape[:2]
    for e in obs_entities:
        box = parse_box(e.get("box_2d", e.get("box")), box_format)
        e["_box"], e["_sig"] = box, None
        if box:
            px = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
            e["_sig"] = identity.signature(frame, px)
        # Motion observed across the burst is a first-class attribute — it's
        # both discriminative ("walking left") and behaviorally meaningful.
        if e.get("motion") and str(e["motion"]).lower() not in ("static", "none", ""):
            attrs = e.get("attributes") or []
            e["attributes"] = list(attrs) + [f"motion: {e['motion']}"]

    # Identity assignment is deterministic code, not the LLM.
    tracked = world.tracked()
    known_ids = {ent.id for ent in tracked}
    attrs_before = {ent.id: tuple(ent.attributes) for ent in tracked}
    matches = identity.match(obs_entities, tracked, now)
    update = world.merge(observation, matches)
    events = world.apply_update(update, now, max_gap=presence_credit)
    resolved = update["entities"]  # apply_update wrote the final ids back

    # A face card per new identity: the first-seen crop, like a guard's log.
    new_thumbs: dict = {}
    for e in resolved:
        if e.get("id") not in known_ids and e.get("_box"):
            b = e["_box"]
            crop = frame[int(b[1] * h):int(b[3] * h), int(b[0] * w):int(b[2] * w)]
            if crop.size:
                ok, buf = cv2.imencode(".jpg", crop)
                if ok:
                    new_thumbs[e["id"]] = buf.tobytes()
                    if thumb_dir:
                        os.makedirs(thumb_dir, exist_ok=True)
                        with open(os.path.join(thumb_dir, f"{e['id']}.jpg"), "wb") as fh:
                            fh.write(buf.tobytes())

    # REASON call — only when there's something to reason about.
    attrs_changed = any(
        tuple(str(a) for a in e.get("attributes", []) or []) != attrs_before.get(e.get("id"))
        for e in resolved
        if e.get("id") in attrs_before
    )
    rule_verdict = None
    if rule or events or attrs_changed:
        try:
            reply = extract_json(
                backend.generate(world.build_reason_prompt(update, events, rule, now))
            ) or {}
        except Exception:
            reply = {}
        events = events + world.add_events(reply.get("events"), now)
        if rule and isinstance(reply.get("rule"), dict):
            r = reply["rule"]
            rule_verdict = Verdict(
                bool(r.get("triggered", False)),
                float(r.get("confidence", 0.0) or 0.0),
                str(r.get("reason", "")),
            )

    world.save()  # cheap; keeps state.json crash-safe
    present, total = world.counts()
    return CycleResult(
        ok=True,
        scene=world.scene,
        events=events,
        rule_verdict=rule_verdict,
        boxes=[
            {"id": e["id"], "label": e.get("label", ""), "box": e["_box"]}
            for e in resolved
            if e.get("_box")
        ],
        new_thumbs=new_thumbs,
        present=present,
        total=total,
        latency=time.monotonic() - t0,
    )
