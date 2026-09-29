# Real-time VLM monitor — POC

Point it at a video feed, describe in plain English what you care about, get
notified when it happens.

Two modes, same pluggable VLM backends (local Ollama or cloud Gemini), no
detectors, no training, no per-rule engineering:

- **Phase 1 — stateless monitor** (`run.py`): sample a frame, ask a
  vision-language model whether your rule is true, debounce, notify.
  Sub-second detection with a worker pool.
- **Phase 2 — world model** (`world_run.py`): a persistent, time-series
  understanding of the scene. Entities keep stable identities across exits and
  re-entries (object permanence), every change lands on an event timeline, the
  model *infers* what happened off-screen, and you can ask it questions —
  even across process restarts.

```
Phase 1:  frame ──> VLM: "is <rule> true?" ──> debounce ──> alert + evidence

Phase 2:  frame burst ──> OBSERVE (VLM, memory-blind: entities + boxes + motion)
                              │
                              v
                          MATCH (code: appearance signatures + motion continuity)
                              │
                              v
                          REASON (LLM + memory: events, inferences, rules)
                              │
                              ├── entity registry (stable ids, dwell stats, thumbnails)
                              ├── event timeline (+ inferences, + flagged uncertainty)
                              └── temporal/behavioral rules + Q&A
```

## Setup

```bash
pip install -r requirements.txt
```

**Local backend (default, no API key):** install [Ollama](https://ollama.com) and pull a vision model:

```bash
ollama pull qwen2.5vl      # recommended; or: ollama pull moondream
```

**Cloud backend (Gemini):**

```bash
export GEMINI_API_KEY=your-key-here
```

## Run

```bash
# Webcam + local VLM
python run.py --rule "a person is not wearing a hard hat"

# A sample video + Gemini
python run.py --source sample.mp4 --backend gemini \
    --rule "a red mug is off-center on the conveyor belt"

# An RTSP camera, only checking when something moves
python run.py --source "rtsp://user:pass@camera-ip/stream" \
    --rule "the door is left open" --motion --display
```

When the rule holds for `--consecutive` checks in a row, you get a console
alert, a macOS notification, and a saved frame in `evidence/`.

## Run the world model (Phase 2)

```bash
# Webcam + Gemini; walk in and out of frame and watch it keep track of you
python world_run.py --backend gemini --model gemini-3.1-flash-lite --display

# Temporal rules that are impossible per-frame (they reference the past):
python world_run.py --backend gemini --rule "the person has been gone for more than a minute"
```

While it runs, type in the terminal:

- `? where is the red mug` — ask the world model anything; it answers from its
  memory, with times, and distinguishes what it observed from what it inferred.
- `the door is left open` + Enter — set/replace the watch rule live (blank line
  clears it).

State persists in `--world-dir` (default `world/`): `state.json` is the entity
registry, `timeline.jsonl` the append-only event log. Restarting resumes the
same world — someone returning an hour later is re-matched to their old
identity. Start over with `--fresh`.

**How it works.** Each cycle has three stages:

1. **Observe** (VLM, memory-blind): describes the entities in a short *burst*
   of frames (`--burst`, default 3) — label, discriminative description,
   attributes, bounding box, and motion across the burst. It's deliberately
   memory-blind: when a small VLM sees its own memory it anchors on it and
   confirms it instead of looking (it kept "seeing" objects that had left).
   Boxes are requested in the backend's native grounding convention
   (Gemini: `box_2d` y-first) — foreign formats measurably degrade accuracy.
2. **Match** ([identity.py](identity.py), pure code — *not* an LLM): assigns
   each observation a stable id by fusing an appearance signature (OpenCV
   color+texture histogram of the box crop) with motion continuity (predicted
   position from the last sighting), solved as an assignment problem.
   Position is what keeps two people in identical clothes separate; appearance
   is what re-identifies someone who left and came back. Matches won by a slim
   margin are logged as `identity_uncertain` events — doubt is recorded, not
   papered over. Text descriptions play no part, so wording drift can't split
   identities. Swap `identity.signature()` for a CLIP/OSNet embedding when you
   want stronger re-ID; everything else stays.
3. **Reason** (LLM + memory, skipped when nothing changed): given the resolved
   changes and per-entity dwell/movement stats, emits state-change, action,
   behavior, and inference events, and evaluates the rule. This is where
   "e3 has been in the store 25 minutes, circling the same shelves, and has
   picked nothing up" comes from.

Enter/exit events are derived in code by diffing present-sets, so they can't
be hallucinated. Every entity keeps dwell time, a movement trail, and a
first-seen thumbnail (`world/entities/e7.jpg`) — the guard's log book. Updates
are serial by design (each cycle must see the state the previous one
produced), so world cadence is ~1.5–3 s; keep `run.py` for sub-second
stateless alerting.

**Cost.** A static scene costs nothing: a ~50µs frame-diff gate (on by
default; `--no-motion-gate` to disable) skips VLM cycles until something
moves, and motion trips it on the very next poll, so event latency is
unchanged. A heartbeat cycle still runs every `--heartbeat` seconds (60 by
default, `--rule-heartbeat`, 15, while a rule is active) so time-based
reasoning — dwell stats, "gone for more than a minute" — keeps advancing.
Dwell time is credited across gated gaps: a skipped stretch verifiably had no
change, so presence spans it. On exit the monitor prints exact token usage
and cost (from the API's own usage metadata) plus how many static cycles the
gate saved. Typical cameras land at $1–3/day on Gemini Flash-Lite instead of
~$29/day ungated; the web app applies the same gate per session.

## Key flags

| Flag | Default | What it does |
|---|---|---|
| `--rule` | (required) | The plain-English condition to watch for |
| `--source` | `0` | Webcam index, file path, or RTSP/HTTP URL |
| `--backend` | `ollama` | `ollama` (local) or `gemini` (cloud) |
| `--model` | backend default | Override the model (e.g. `moondream`, `gemini-2.5-flash`) |
| `--interval` | `1.0` | Seconds between VLM checks |
| `--consecutive` | `2` | Positive checks in a row before alerting (kills single-frame blips) |
| `--cooldown` | `30` | Min seconds between repeat alerts |
| `--motion` | off | Skip checks on near-static frames (saves VLM calls) |
| `--display` | off | Show the video window (press `q` to quit) |

## Notes & limits (it's a POC)

- **Latency budget is seconds, not milliseconds** — `--interval` + `--consecutive`
  set how fast an alert lands (default ~2s). That's the right trade for alerting.
- **Precise spatial rules** ("off-center", "how many") are where pure VLMs are
  weakest — they're the first thing Phase 2's detector/geometry layer will take over.
- **Video files** are read as fast as they decode; the `--interval` gate samples
  by wall-clock, so files race ahead of real time. Fine for testing.

## GAVI — the web app

**Live: https://gavi-production.up.railway.app** (Railway project `gavi`;
redeploy with `railway up --detach` from this directory).

```bash
make web        # then open http://localhost:8000
```

The landing page for **GAVI (General Artificial Visual Intelligence)**:
a generated CCTV-style hero video with a live tracking-HUD overlay, three
preset demos (object permanence / identical outfits / loitering — AI-generated
clips in `web/assets/`, streamed server-side through the real pipeline with
play/stop controls and live Q&A), an animated world-model explainer, and a
try-it-yourself webcam demo. Everything on screen is computed live by
engine.py — no recordings of results. Each visitor gets an isolated,
ephemeral world (temp dir, deleted on disconnect, 15-minute cap).
Server: [app.py](app.py) · page: [web/index.html](web/index.html). To demo to
someone on your network, open `http://<your-ip>:8000` — note browsers require
HTTPS for camera access on non-localhost origins, so put it behind a TLS proxy
(e.g. `caddy` or an ngrok/Tailscale HTTPS URL) for remote viewers.

## Extending

Add a backend by subclassing `VLMBackend` in [backends.py](backends.py) and
registering it in `make_backend()`. The contract is two methods:
`check(jpeg_bytes, rule) -> Verdict(triggered, confidence, reason)` for the
phase-1 monitor, and `generate(prompt, images=None, json_mode=True) -> str`
(single JPEG or a burst list) for the world model. Set `box_format` if the
model's native grounding order isn't `[x0,y0,x1,y1]`.

The world-model brain lives in [world.py](world.py) (entity registry, event
timeline, prompts, persistence) and [identity.py](identity.py) (appearance +
motion matching). The runtime loop is [world_run.py](world_run.py).
