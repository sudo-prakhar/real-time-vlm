"""Persistent world model for the VLM monitor (Phase 2).

The phase-1 monitor is stateless: every frame is judged in isolation. This
module adds the memory. A WorldModel holds:

  - an entity registry: everything the system has ever tracked, each with a
    stable id, description, attributes, and present/absent status. Entities
    that leave the frame are NOT dropped — they stay in the registry with a
    last_seen time, which is what gives the system object permanence: when
    someone walks back in, the VLM re-matches them to their existing id.
  - an event timeline: an append-only log of entries/exits, state changes,
    actions, and *inferences* (the model's best guess at what happened while
    something was out of view — the "fill in the gaps" part).

The update cycle (driven by world_run.py) is two calls per frame:

  A. OBSERVE (vision, memory-BLIND): "describe the entities you see", with
     bounding boxes. The world state is deliberately kept out of this prompt —
     when a small VLM sees its own memory it anchors on it and confirms it
     instead of looking (measured: it kept "seeing" an object for 4+ frames
     after it left, and kept missing one after it returned). Perception must
     be stateless.
  B. MATCH (pure code, identity.py): assign each observation a stable entity
     id using appearance signatures + motion continuity. Not an LLM — text
     descriptions merge two people in the same clothes; position and pixels
     don't. Slim-margin wins are flagged as identity_uncertain events.
  C. REASON (text-only LLM, sees memory): given the resolved changes, emit
     state-change/action/inference events and evaluate the watch rule.
     Skipped when nothing changed and no rule is set.

apply_update() folds the merged result in and accumulates per-entity behavior
(dwell time, movement trail) for loitering-style rules. Enter/exit events are
derived here in code (diffing present-sets), not asked of the model, so they
can't be hallucinated.

State survives restarts: a snapshot goes to <dir>/state.json and every event
is appended to <dir>/timeline.jsonl as it happens.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class Entity:
    id: str
    label: str  # coarse class: "person", "dog", "mug"
    description: str  # discriminative: "man in a blue t-shirt with glasses"
    attributes: list[str] = field(default_factory=list)
    location: str = ""  # coarse, in words: "left third, midground"
    present: bool = False
    first_seen: float = 0.0  # epoch seconds
    last_seen: float = 0.0
    sightings: int = 0
    box: list[float] | None = None  # last known [x0,y0,x1,y1], normalized 0..1
    signatures: list[list[float]] = field(default_factory=list)  # appearance gallery
    total_visible: float = 0.0  # accumulated on-screen seconds
    trail: list[list[float]] = field(default_factory=list)  # [t, cx, cy] path history
    attr_since: dict = field(default_factory=dict)  # normalized attr -> start epoch

    GALLERY_SIZE = 8
    TRAIL_SIZE = 400

    def update_attributes(
        self, new_attrs: list[str], now: float, max_gap: float = 5.0
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Carry per-attribute start times across cycles so held states have a
        measured duration. Returns (started, ended-with-held-seconds).

        Matching is fuzzy (token overlap) because the VLM rephrases the same
        state between frames; a gap longer than `max_gap` breaks continuity —
        under a motion gate a gated gap is verifiably static, so the runtime
        passes its heartbeat and states persist through it.
        """
        continuous = self.present and (now - self.last_seen) <= max_gap
        old = dict(self.attr_since) if continuous else {}
        matched_old: set[str] = set()
        fresh: dict[str, float] = {}
        for a in new_attrs:
            key = _norm_attr(a)
            if key in fresh:
                continue
            best, best_sim = None, ATTR_CARRY_SIM
            for ok in old:
                if ok in matched_old:
                    continue
                sim = 1.0 if ok == key else _attr_sim(key, ok)
                if sim >= best_sim:
                    best, best_sim = ok, sim
            if best is not None:
                fresh[key] = old[best]
                matched_old.add(best)
            else:
                fresh[key] = now
        started = [a for a in new_attrs if fresh.get(_norm_attr(a)) == now]
        ended = [(ok, now - old[ok]) for ok in old if ok not in matched_old]
        self.attr_since = fresh
        return started, ended

    def held_for(self, attr: str, now: float) -> float:
        t0 = self.attr_since.get(_norm_attr(attr))
        return (now - t0) if t0 else 0.0

    def record_sighting(
        self, now: float, box: list[float] | None, sig: list[float] | None,
        max_gap: float = 5.0,
    ) -> None:
        if self.sightings and self.present:
            # Credit on-screen time only across gaps of continuous presence.
            # The default covers normal cycle spacing; a motion-gated runtime
            # passes its heartbeat instead — a gated gap means the scene
            # verifiably did not change, so the entity was there throughout.
            self.total_visible += min(now - self.last_seen, max_gap)
        if box:
            self.box = box
            self.trail.append([now, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            if len(self.trail) > self.TRAIL_SIZE:
                self.trail = self.trail[::2]  # decimate: keep horizon, halve density
        if sig:
            self.signatures.append(sig)
            if len(self.signatures) > self.GALLERY_SIZE:
                self.signatures.pop(0)

    def zones_visited(self, now: float, window: float = 600.0) -> int:
        """Distinct cells of a 3x3 frame grid crossed in the last `window` secs."""
        return len({
            (min(int(cx * 3), 2), min(int(cy * 3), 2))
            for t, cx, cy in self.trail
            if now - t <= window
        })


@dataclass
class Event:
    t: float  # epoch seconds
    type: str  # entered | exited | state_change | action | inference
    text: str
    entity_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0


def _dur(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _ago(seconds: float) -> str:
    return f"{_dur(seconds)} ago"


# --- attribute persistence -----------------------------------------------
# "held the mug for 10 seconds" needs a measured clock per attribute, not the
# model's impression. Attributes are matched across cycles fuzzily because the
# VLM rephrases the same state ("holding a red mug" / "holding red mug").

_ATTR_STOPWORDS = {"a", "an", "the", "is", "of", "in", "on", "at", "with", "and", "their", "his", "her"}


def _norm_attr(a: str) -> str:
    return " ".join(str(a).lower().split())


def _attr_tokens(a: str) -> frozenset:
    return frozenset(t for t in _norm_attr(a).replace(",", " ").split() if t not in _ATTR_STOPWORDS)


def _attr_sim(a: str, b: str) -> float:
    ta, tb = _attr_tokens(a), _attr_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


ATTR_CARRY_SIM = 0.3  # rephrasings usually share the key nouns/verbs


OBSERVE_PROMPT = """\
{intro} Report ONLY what is actually visible — you have no other context, and
nothing is implied to be present.

List EVERY salient entity {visible_in}: people, animals, vehicles,
objects being held/moved, and notable scene objects (door, laptop, mug...).
At most 10 — prefer the ones that matter. For each give:
- "label": coarse class ("person", "dog", "mug")
- "description": discriminative, stable across frames, 8 words max
  ("man in a blue t-shirt with glasses" — identity, not pose)
- "attributes": current changeable state (["holding a red mug", "sitting"]),
  at most 3, [] when nothing notable. REUSE the exact same wording for a state
  that continues across frames — the tracker times how long each state lasts
- "location": coarse position, 2-4 words ("left third, midground")
- "{box_key}": tight bounding box {box_desc} as integers 0-1000, scaled to the image{box_last}
{motion_field}
Respond with strict JSON and nothing else:
{{"scene": "<one short sentence (max 12 words) describing the {final}>",
 "entities": [{{"label": "...", "description": "...", "attributes": ["..."], "location": "...",
               "{box_key}": {box_desc_short}{motion_schema}}}]}}"""

BURST_INTRO = (
    "You are looking at {n} consecutive frames from the SAME fixed camera, "
    "oldest first, spanning about {span:.1f} seconds. Describe them for an "
    "object-tracking system."
)
SINGLE_INTRO = "Describe this video frame for an object-tracking system."
MOTION_FIELD = (
    '- "motion": how the entity moved across the frames '
    '("walking left", "static", "just picked up the bag") — use the frame '
    "sequence; this is what tells identical-looking entities apart\n"
)

BOX_FORMATS = {
    # Gemini's grounding was trained on box_2d/[ymin,xmin,ymax,xmax]; asking in
    # a foreign order measurably degrades box quality.
    "yxyx": ("box_2d", "[y_min, x_min, y_max, x_max]", "[y_min, x_min, y_max, x_max]"),
    "xyxy": ("box", "[x_min, y_min, x_max, y_max]", "[x_min, y_min, x_max, y_max]"),
}

REASON_PROMPT = """\
You are the reasoning module of a world model watching a video feed. A vision
module observed the current frame and a visual matcher already resolved every
visible entity to a stable id — do NOT second-guess identities. Your job is to
notice what the changes MEAN.

CURRENT TIME: {now}

TRACKED ENTITIES (memory, with per-entity dwell/movement stats):
{state}

RECENT EVENTS:
{events}

THIS CYCLE:
{cycle}

Tasks:
1. Report "events" ONLY for: state changes, actions/interactions, notable
   behavior patterns (using the dwell/movement stats — e.g. someone present a
   long time, circling the same areas, without purposeful activity), and
   inferences. Do NOT report entering/leaving the frame — already derived.
   Do NOT restate state changes already listed under THIS CYCLE.
   An "inference" fills a gap: when an entity returns after an absence with
   something changed, say what most likely happened off-screen (e.g.
   "returned without the mug — likely set it down in another room").
   Set confidence honestly; inferences are guesses, not observations.
   No event is a fine answer — do not invent one per cycle.
   Keep each event text under 15 words.
   Durations in the state like "(14s so far)" are MEASURED by the tracker —
   trust them for any "has been X for longer than N" judgement.
{rule_clause}

Respond with strict JSON and nothing else:
{{
  "events": [
    {{"type": "state_change|action|behavior|inference", "text": "<what happened>",
      "entity_ids": ["<id>"], "confidence": <0.0-1.0>}}
  ]{rule_schema}
}}"""

RULE_CLAUSE = """\
2. Also evaluate this rule against the CURRENT world state (memory + this
   cycle) — it may reference the past ("has been gone for...", "has been
   loitering for..."), behavior stats, or absent entities:
   RULE: "{rule}"
   If the rule requires a DURATION ("for more than N seconds/minutes"),
   trigger ONLY when a MEASURED duration in the state ("Xs so far") or the
   timeline meets it. Being in the state right now is NOT enough — say how
   long it has measurably held in your reason.
"""

RULE_SCHEMA = ',\n  "rule": {"triggered": <true|false>, "confidence": <0.0-1.0>, "reason": "<short>"}'

ANSWER_PROMPT = """\
You maintain a world model of a video feed: a registry of tracked entities and
a timeline of observed events and inferences. Answer the user's question from
this memory. Be concrete about times ("about 2 minutes ago"). If the memory
genuinely doesn't contain the answer, say so. Distinguish what was observed
from what was inferred.

CURRENT TIME: {now}

TRACKED ENTITIES:
{state}

EVENT TIMELINE (oldest first):
{events}

QUESTION: {question}

Answer in 1-3 sentences of plain text (no JSON)."""


class WorldModel:
    """Entity registry + event timeline with disk persistence. Thread-safe."""

    def __init__(self, directory: str = "world", resume: bool = True):
        self.dir = directory
        self._lock = threading.Lock()
        self.entities: dict[str, Entity] = {}
        self.events: list[Event] = []
        self.scene = ""
        self._next_id = 1
        os.makedirs(self.dir, exist_ok=True)
        if resume:
            self._load()

    # -- persistence -----------------------------------------------------

    @property
    def _state_path(self) -> str:
        return os.path.join(self.dir, "state.json")

    @property
    def _timeline_path(self) -> str:
        return os.path.join(self.dir, "timeline.jsonl")

    def _load(self) -> None:
        if os.path.exists(self._state_path):
            with open(self._state_path) as f:
                data = json.load(f)
            self.entities = {e["id"]: Entity(**e) for e in data.get("entities", [])}
            self._next_id = data.get("next_id", len(self.entities) + 1)
            # A restart means we haven't looked in a while — trust nothing as present.
            for e in self.entities.values():
                e.present = False
        if os.path.exists(self._timeline_path):
            with open(self._timeline_path) as f:
                self.events = [Event(**json.loads(line)) for line in f if line.strip()]

    def save(self) -> None:
        with self._lock:
            data = {
                "next_id": self._next_id,
                "entities": [asdict(e) for e in self.entities.values()],
            }
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self._state_path)

    def _append_event(self, ev: Event) -> None:
        self.events.append(ev)
        with open(self._timeline_path, "a") as f:
            f.write(json.dumps(asdict(ev)) + "\n")

    # -- prompt rendering ------------------------------------------------

    def render_state(self, now: float, max_absent: int = 20) -> str:
        """Compact one-entity-per-line view for the update prompt."""
        present = [e for e in self.entities.values() if e.present]
        absent = sorted(
            (e for e in self.entities.values() if not e.present),
            key=lambda e: e.last_seen,
            reverse=True,
        )[:max_absent]
        lines = []
        for e in present:
            # attributes carry their measured duration — the ground truth for
            # "has been X for longer than N seconds" rules
            parts = []
            for a in e.attributes:
                held = e.held_for(a, now)
                parts.append(f"{a} ({_dur(held)} so far)" if held >= 3 else a)
            attrs = f" [{', '.join(parts)}]" if parts else ""
            stats = (
                f" — first seen {_ago(now - e.first_seen)}, on screen {_dur(e.total_visible)} total"
            )
            zones = e.zones_visited(now)
            if zones:
                stats += f", moved through {zones}/9 frame areas in the last 10m"
            lines.append(
                f'- {e.id} ({e.label}, IN FRAME): "{e.description}"{attrs} at {e.location}{stats}'
            )
        for e in absent:
            attrs = f" [{', '.join(e.attributes)}]" if e.attributes else ""
            lines.append(
                f'- {e.id} ({e.label}, OUT OF FRAME, last seen {_ago(now - e.last_seen)}, '
                f'was on screen {_dur(e.total_visible)} total): "{e.description}"{attrs}'
            )
        return "\n".join(lines) if lines else "(none yet — this is a fresh world)"

    def render_events(self, now: float, n: int = 15) -> str:
        lines = [
            f"- [{_ago(now - ev.t)}] {ev.type}: {ev.text}"
            + (f" (confidence {ev.confidence:.0%})" if ev.type == "inference" else "")
            for ev in self.events[-n:]
        ]
        return "\n".join(lines) if lines else "(none yet)"

    def build_observe_prompt(self, box_format: str = "xyxy", burst: int = 1, span: float = 0.0) -> str:
        # Deliberately state-free — see module docstring.
        key, desc, short = BOX_FORMATS.get(box_format, BOX_FORMATS["xyxy"])
        multi = burst > 1
        return OBSERVE_PROMPT.format(
            intro=BURST_INTRO.format(n=burst, span=span) if multi else SINGLE_INTRO,
            visible_in="visible in the FINAL frame" if multi else "currently visible",
            box_key=key,
            box_desc=desc,
            box_desc_short=short,
            box_last=" — on the FINAL frame" if multi else "",
            motion_field=MOTION_FIELD if multi else "",
            motion_schema=', "motion": "..."' if multi else "",
            final="final frame" if multi else "frame",
        )

    def build_reason_prompt(
        self,
        update: dict,
        cycle_events: list[Event],
        rule: str | None,
        now: float | None = None,
    ) -> str:
        now = now or time.time()
        cycle = [f"scene: {update.get('scene', '')}"]
        visible = [
            f'  {e.get("id")}: "{e.get("description", "")}"'
            + (f' [{", ".join(str(a) for a in e.get("attributes", []))}]' if e.get("attributes") else "")
            + f' at {e.get("location", "?")}'
            for e in update.get("entities", []) or []
        ]
        cycle.append("visible now:" if visible else "visible now: (nothing salient)")
        cycle.extend(visible)
        if cycle_events:
            cycle.append("changes this cycle:")
            cycle.extend(f"  - {ev.type}: {ev.text}" for ev in cycle_events)
        with self._lock:
            state = self.render_state(now)
            events = self.render_events(now)
        return REASON_PROMPT.format(
            now=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            state=state,
            events=events,
            cycle="\n".join(cycle),
            rule_clause=RULE_CLAUSE.format(rule=rule) if rule else "",
            rule_schema=RULE_SCHEMA if rule else "",
        )

    @staticmethod
    def merge(observation: dict, matches: list[dict]) -> dict:
        """Attach the matcher's id assignments to the observation's entities.

        The observation is authoritative for WHAT is visible; identity.match()
        decided WHO each one is. Entities left with id=None become new.
        """
        merged = []
        for i, e in enumerate(observation.get("entities", []) or []):
            if not isinstance(e, dict):
                continue
            m = matches[i] if i < len(matches) else {}
            merged.append({
                **e,
                "id": m.get("id"),
                "_score": float(m.get("score", 0.0) or 0.0),
                "_uncertain": bool(m.get("uncertain")),
                "_runner_up": m.get("runner_up"),
            })
        return {"scene": observation.get("scene", ""), "entities": merged}

    def build_answer_prompt(self, question: str, now: float | None = None) -> str:
        now = now or time.time()
        with self._lock:
            state = self.render_state(now, max_absent=50)
            events = self.render_events(now, n=80)
        return ANSWER_PROMPT.format(
            now=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            state=state,
            events=events,
            question=question,
        )

    # -- update cycle ----------------------------------------------------

    def apply_update(
        self, update: dict, now: float | None = None, max_gap: float = 5.0
    ) -> list[Event]:
        """Fold one VLM update into the world. Returns the new events.

        `max_gap` is the longest sighting gap that still counts as continuous
        presence for dwell stats (see Entity.record_sighting)."""
        now = now or time.time()
        new_events: list[Event] = []
        with self._lock:
            self.scene = str(update.get("scene", "") or self.scene)

            seen_ids: set[str] = set()
            for raw in update.get("entities", []) or []:
                if not isinstance(raw, dict):
                    continue
                eid = raw.get("id")
                label = str(raw.get("label", "thing"))
                desc = str(raw.get("description", label))
                attrs = [str(a) for a in raw.get("attributes", []) or []]
                loc = str(raw.get("location", ""))
                box = raw.get("_box")  # normalized, set by the runtime
                sig = raw.get("_sig")

                ent = self.entities.get(eid) if isinstance(eid, str) else None
                if ent is None:  # matcher found no convincing owner -> new identity
                    ent = Entity(
                        id=f"e{self._next_id}", label=label, description=desc,
                        attributes=attrs, location=loc,
                        present=True, first_seen=now, last_seen=now, sightings=0,
                    )
                    self._next_id += 1
                    self.entities[ent.id] = ent
                    new_events.append(Event(now, "entered", f"{ent.id} ({desc}) entered the frame", [ent.id]))
                    ent.update_attributes(attrs, now, max_gap)  # start the clocks
                else:
                    if not ent.present:  # object permanence pay-off: a re-match
                        new_events.append(Event(
                            now, "entered",
                            f"{ent.id} ({ent.description}) re-entered after "
                            f"{_dur(now - ent.last_seen)} away", [ent.id],
                        ))
                    # Measured state history: per-attribute clocks + derived
                    # state_change events — this is what makes rules like
                    # "held the mug for over 10 seconds" answerable.
                    started, ended = ent.update_attributes(attrs, now, max_gap)
                    if ent.sightings > 0:
                        for a in started:
                            if not _norm_attr(a).startswith("motion:"):
                                new_events.append(Event(
                                    now, "state_change", f"{ent.id} now: {a}", [ent.id]))
                        for a, held in ended:
                            if not a.startswith("motion:") and held >= 2.0:
                                new_events.append(Event(
                                    now, "state_change",
                                    f"{ent.id} no longer: {a} — lasted {_dur(held)}", [ent.id]))
                    ent.label, ent.description = label, desc
                    ent.attributes, ent.location = attrs, loc
                if raw.get("_uncertain") and raw.get("_runner_up"):
                    new_events.append(Event(
                        now, "identity_uncertain",
                        f"{ent.id} matched by a slim margin — could also be "
                        f"{raw['_runner_up']}", [ent.id, str(raw["_runner_up"])], 0.5,
                    ))
                # Memory hygiene: learn appearance AND position from founding
                # sightings and confident matches only — a jittery box yields a
                # garbage crop and a wrong location, and either one poisons
                # every future comparison against this entity.
                is_new = ent.sightings == 0
                confident = is_new or float(raw.get("_score", 0.0)) >= 0.55
                ent.record_sighting(
                    now,
                    box if confident and isinstance(box, list) else None,
                    sig if confident else None,
                    max_gap=max_gap,
                )
                ent.present, ent.last_seen = True, now
                ent.sightings += 1
                raw["id"] = ent.id  # write back so the runtime can crop thumbnails
                seen_ids.add(ent.id)

            # Anything we believed present but the model no longer lists → left frame.
            for ent in self.entities.values():
                if ent.present and ent.id not in seen_ids:
                    ent.present = False
                    new_events.append(Event(
                        now, "exited",
                        f"{ent.id} ({ent.description}) left the frame — still tracked", [ent.id],
                    ))

            for ev in new_events:
                self._append_event(ev)
        return new_events

    def add_events(self, raw_events: list, now: float | None = None) -> list[Event]:
        """Fold the reasoning module's events (state changes, inferences) in."""
        now = now or time.time()
        added = []
        with self._lock:
            for raw in raw_events or []:
                if not isinstance(raw, dict) or not raw.get("text"):
                    continue
                etype = str(raw.get("type", "state_change"))
                if etype in ("entered", "exited"):
                    continue  # derived in apply_update; don't let the model duplicate them
                ev = Event(
                    now, etype, str(raw["text"]),
                    [str(i) for i in raw.get("entity_ids", []) or []],
                    float(raw.get("confidence", 1.0) or 1.0),
                )
                self._append_event(ev)
                added.append(ev)
        return added

    # -- introspection ---------------------------------------------------

    def counts(self) -> tuple[int, int]:
        with self._lock:
            present = sum(e.present for e in self.entities.values())
            return present, len(self.entities)

    def tracked(self) -> list[Entity]:
        """Snapshot of tracked entities for the matcher (read-only use)."""
        with self._lock:
            return list(self.entities.values())

    def present_boxes(self) -> list[tuple[str, list[float]]]:
        """(id, normalized box) for in-frame entities — for display overlays."""
        with self._lock:
            return [(e.id, e.box) for e in self.entities.values() if e.present and e.box]
