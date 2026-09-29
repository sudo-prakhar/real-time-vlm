"""Identity assignment for the world model: who is each observed entity?

The LLM reconciler used to assign ids by comparing text descriptions — which
merges two people wearing the same clothes. This module replaces that with a
deterministic matcher built on the two cues a human guard actually uses:

  - MOTION CONTINUITY: an entity can't teleport. Each tracked entity's next
    position is predicted from its last box + velocity; nearby observations
    score high. This is what keeps two identically-dressed people separate —
    appearance is useless there, position is decisive.
  - APPEARANCE: a compact visual signature computed from the entity's crop
    (HSV color histogram + coarse luminance patch — pure OpenCV, no model
    download). A gallery of recent signatures per entity handles pose/light
    drift. This is what re-identifies someone who left and came back.

Scores are fused and solved as an assignment problem (Hungarian via scipy,
greedy fallback). Below-threshold observations become NEW entities; a win by
a slim margin over the runner-up is flagged uncertain so the timeline records
doubt instead of silently guessing.

Upgrade path: signature() is the only vision code. Swap it for a CLIP/OSNet
embedding (e.g. open_clip, if installed) and everything else — gallery,
motion model, assignment — stays identical.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# Fused-score weights and thresholds (scores all live in [0, 1]).
W_APPEARANCE = 0.6
W_SPATIAL = 0.4
MATCH_THRESHOLD = 0.35  # best score below this -> new entity
UNCERTAIN_MARGIN = 0.10  # winner beats runner-up by less than this -> flag it
SPATIAL_HORIZON = 20.0  # seconds after which position stops being evidence
SPATIAL_OVERRIDE = 0.75  # position this consistent wins even if the crop looked wrong
CONFIDENT_SIG = 0.55  # gallery only learns from matches at least this strong
GALLERY_SIZE = 8  # signatures kept per entity


def signature(frame_bgr: np.ndarray, box_px: tuple[int, int, int, int]) -> list[float] | None:
    """Appearance signature of a crop: HSV histogram + 8x8 luminance patch.

    Computed on the central 70% of the box — VLM boxes are jittery, and the
    center survives a loose or shifted box far better than the edges do.
    Returns a plain list (JSON-serializable for state.json), L2-normalized.
    """
    x0, y0, x1, y1 = box_px
    mx, my = int((x1 - x0) * 0.15), int((y1 - y0) * 0.15)
    x0, y0, x1, y1 = x0 + mx, y0 + my, x1 - mx, y1 - my
    h, w = frame_bgr.shape[:2]
    x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    import cv2

    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= hist.sum() + 1e-9
    patch = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (8, 8)).flatten() / 255.0
    vec = np.concatenate([hist * 2.0, patch * 0.5])  # color dominates, texture assists
    n = np.linalg.norm(vec)
    return (vec / n).tolist() if n > 0 else None


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    if va.shape != vb.shape:
        return 0.0
    return float(np.dot(va, vb))  # signatures are already unit-norm


def appearance_score(sig: list[float] | None, gallery: list[list[float]]) -> float | None:
    """Best cosine match against the entity's signature gallery, or None if unknowable."""
    if sig is None or not gallery:
        return None
    return max(_cosine(sig, g) for g in gallery)


def spatial_score(
    box: tuple[float, float, float, float] | None,
    last_box: list[float] | None,
    gap_secs: float,
) -> float | None:
    """How consistent this position is with where the entity was last seen.

    Boxes are normalized [x0,y0,x1,y1] in 0..1. Returns None when position
    carries no evidence (no boxes, or the entity has been gone too long).
    """
    if box is None or not last_box or gap_secs > SPATIAL_HORIZON:
        return None
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    lx, ly = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
    dist = float(np.hypot(cx - lx, cy - ly))  # 0..~1.4 across the frame
    # Allow ~15% of the frame of drift per second away, floor at a tight radius.
    reach = max(0.08, 0.15 * max(gap_secs, 0.5))
    score = max(0.0, 1.0 - dist / reach)
    # Position is strong evidence right after a sighting, weaker as time passes.
    return score * max(0.0, 1.0 - gap_secs / SPATIAL_HORIZON)


PERSON_WORDS = {
    "person", "man", "woman", "child", "boy", "girl", "human", "people",
    "lady", "guy", "pedestrian", "worker", "customer",
}


def _is_person(label: str) -> bool:
    return any(w in label for w in PERSON_WORDS)


def label_affinity(a: str, b: str) -> float:
    """How much two open-vocabulary labels support being the same thing.

    A VLM's label for the same object drifts between synonyms ("circle" /
    "dot" / "ball"), so mismatch can't be a hard gate — measured live, that
    minted a fresh identity on every wording change. Same/overlapping labels
    are full support; differing labels merely discount (appearance + position
    decide); only the person/non-person boundary is a hard zero.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or (a and b and (a in b or b in a)):
        return 1.0
    if _is_person(a) != _is_person(b):
        return 0.0  # a mug never becomes a person
    return 0.7


def labels_compatible(a: str, b: str) -> bool:
    return label_affinity(a, b) > 0.0


def score_pair(obs: dict, entity, now: float) -> float:
    """Fused match score between one observation and one tracked entity.

    obs carries "_sig" (signature or None) and "_box" (normalized or None),
    set by the caller. Missing evidence renormalizes onto what's available.
    """
    affinity = label_affinity(str(obs.get("label", "")), entity.label)
    if affinity == 0.0:
        return 0.0
    app = appearance_score(obs.get("_sig"), entity.signatures)
    spa = spatial_score(obs.get("_box"), entity.box, now - entity.last_seen)
    parts = [(W_APPEARANCE, app), (W_SPATIAL, spa)]
    total_w = sum(w for w, s in parts if s is not None)
    if total_w == 0:
        return 0.0  # no visual evidence either way -> can't claim a match
    fused = sum(w * s for w, s in parts if s is not None) / total_w
    # Objects don't teleport: near-perfect positional continuity outranks a bad
    # appearance read (VLM boxes miss sometimes, making the crop garbage).
    if spa is not None and spa >= SPATIAL_OVERRIDE:
        fused = max(fused, spa)
    # Prefer continuity over resurrection: with no positional evidence (long
    # absence), pure appearance must not outbid a present look-alike — else
    # identical twins flip-flop ids whenever one of them is off-screen.
    if spa is None:
        fused *= 0.85
    return fused * affinity


def match(observations: list[dict], entities: list, now: float) -> list[dict]:
    """Assign observations to tracked entities.

    Returns one result per observation:
      {"id": <entity id or None>, "score": float, "uncertain": bool,
       "runner_up": <entity id or None>}
    """
    results = [{"id": None, "score": 0.0, "uncertain": False, "runner_up": None} for _ in observations]
    if not observations or not entities:
        return results

    scores = np.zeros((len(observations), len(entities)))
    for i, obs in enumerate(observations):
        for j, ent in enumerate(entities):
            scores[i, j] = score_pair(obs, ent, now)

    if _HAVE_SCIPY:
        rows, cols = linear_sum_assignment(-scores)
        pairs = list(zip(rows, cols))
    else:  # greedy: best global score first
        pairs, used_i, used_j = [], set(), set()
        for i, j in sorted(
            ((i, j) for i in range(len(observations)) for j in range(len(entities))),
            key=lambda ij: -scores[ij[0], ij[1]],
        ):
            if i not in used_i and j not in used_j:
                pairs.append((i, j))
                used_i.add(i)
                used_j.add(j)

    for i, j in pairs:
        s = scores[i, j]
        if s < MATCH_THRESHOLD:
            continue  # not convincing -> new entity
        row = scores[i].copy()
        row[j] = -1
        runner = int(row.argmax()) if len(entities) > 1 else -1
        margin = s - (row[runner] if runner >= 0 else 0.0)
        results[i] = {
            "id": entities[j].id,
            "score": float(s),
            "uncertain": bool(margin < UNCERTAIN_MARGIN),
            "runner_up": entities[runner].id if runner >= 0 and margin < UNCERTAIN_MARGIN else None,
        }

    # Fallback for observations with NO visual evidence (backend gave no usable
    # box, so no signature either): if exactly one still-unclaimed entity has a
    # compatible label, take it — flagged uncertain — instead of spawning a
    # duplicate identity every cycle.
    claimed = {r["id"] for r in results if r["id"]}
    for i, obs in enumerate(observations):
        if results[i]["id"] or obs.get("_sig") is not None or obs.get("_box") is not None:
            continue
        candidates = [
            e for e in entities
            if e.id not in claimed and labels_compatible(str(obs.get("label", "")), e.label)
        ]
        if len(candidates) == 1:
            results[i] = {"id": candidates[0].id, "score": 0.0, "uncertain": True, "runner_up": None}
            claimed.add(candidates[0].id)
    return results
