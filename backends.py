"""Pluggable VLM backends for the real-time monitor.

Each backend takes a JPEG frame (bytes) + a natural-language rule and returns a
structured Verdict. Add a backend by subclassing VLMBackend and implementing
check(), then register it in make_backend().
"""

from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass


@dataclass
class Verdict:
    triggered: bool
    confidence: float
    reason: str


PROMPT_TEMPLATE = (
    "You are a video-monitoring assistant. Look at the image and evaluate ONLY "
    "this rule:\n\n"
    'RULE: "{rule}"\n\n'
    "Decide whether the rule's condition is currently TRUE in the image.\n"
    "{stance}\n\n"
    "Respond with strict JSON and nothing else:\n"
    '{{"triggered": <true|false>, "confidence": <0.0-1.0>, "reason": "<short explanation>"}}'
)

# How eagerly to trigger. "liberal" suits a cheap first-pass filter feeding a
# smarter second stage — favor recall (catch more, tolerate false positives).
STANCES = {
    "liberal": (
        "This is a coarse FIRST-PASS filter feeding a smarter second stage, so "
        "favor recall over precision: if the condition might be true, or you are "
        "unsure, set triggered=true. Only set triggered=false when the condition "
        "is clearly and obviously absent."
    ),
    "balanced": (
        "Judge whether the condition is actually happening: triggered=true if it "
        "is, false if it isn't."
    ),
    "strict": (
        "Be conservative: set triggered=true only when you are clearly sure the "
        "condition is happening; if you are not reasonably sure, set triggered=false."
    ),
}


def build_prompt(rule: str, sensitivity: str) -> str:
    return PROMPT_TEMPLATE.format(
        rule=rule, stance=STANCES.get(sensitivity, STANCES["balanced"])
    )


def _repair_brackets(text: str) -> str:
    """Fix mismatched closing brackets — e.g. `[1, 2}` -> `[1, 2]`.

    Small models glitch a `}` for a `]` (or vice versa) surprisingly often;
    a stack scan that swaps any closer for whatever is actually open recovers
    the frame instead of dropping it.
    """
    out, stack, in_str, esc = [], [], False, False
    for ch in text:
        if in_str:
            out.append(ch)
            esc = not esc and ch == "\\"
            in_str = in_str and (ch != '"' or esc)
            if ch != "\\":
                esc = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                ch = "}" if stack.pop() == "{" else "]"
            else:
                continue  # stray closer with nothing open — drop it
        out.append(ch)
    out.extend("}" if b == "{" else "]" for b in reversed(stack))  # close leftovers
    return "".join(out)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (tolerates ``` fences/prose)."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    for candidate in (text, _repair_brackets(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None


def _parse_verdict(text: str) -> Verdict:
    """Extract the first JSON object from the model reply and coerce it."""
    data = extract_json(text)
    if data is None:
        return Verdict(False, 0.0, f"unparseable model reply: {(text or '')[:200]!r}")
    return Verdict(
        triggered=bool(data.get("triggered", False)),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        reason=str(data.get("reason", "")),
    )


def _gemini_text(resp) -> str:
    """Concatenate only the text parts of a Gemini response.

    Avoids the google-genai warning that fires when you access resp.text and the
    response also carries non-text parts (e.g. a `thought_signature` from the
    Gemini 3 thinking models). Those parts are metadata we don't need.
    """
    try:
        parts = resp.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return getattr(resp, "text", "") or ""
    return "".join(getattr(p, "text", None) or "" for p in parts)


class VLMBackend:
    name = "base"
    # Native bounding-box convention: "xyxy" = [x0,y0,x1,y1], "yxyx" = [y0,x0,y1,x1]
    # (both 0-1000). Prompting a model in its trained format yields better boxes.
    box_format = "xyxy"

    def check(self, jpeg: bytes, rule: str) -> Verdict:
        raise NotImplementedError

    def generate(self, prompt: str, images: bytes | list[bytes] | None = None, json_mode: bool = True) -> str:
        """One free-form model call: text prompt, optional image(s), optional JSON mode.

        `images` may be a single JPEG or a list (e.g. a burst of consecutive
        frames — multi-image lets the model see motion). The world-model layer
        builds on this; check() stays the narrow rule-verdict path used by the
        phase-1 monitor.
        """
        raise NotImplementedError

    @staticmethod
    def _as_list(images: bytes | list[bytes] | None) -> list[bytes]:
        if images is None:
            return []
        return images if isinstance(images, list) else [images]


class OllamaBackend(VLMBackend):
    """Local, on-device VLM via Ollama. Zero per-frame cost, no API key.

    Setup:  ollama pull qwen2.5vl   (or: ollama pull moondream)
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5vl",
        host: str = "http://localhost:11434",
        sensitivity: str = "liberal",
    ):
        import requests  # local import so a Gemini-only user needn't care

        self._requests = requests
        self.model = model
        self.host = host.rstrip("/")
        self.sensitivity = sensitivity
        self._send_think = True  # auto-disabled if the model rejects the flag

    def _chat(
        self,
        b64s: list[str],
        prompt: str,
        temperature: float = 0.0,
        json_mode: bool = True,
        num_predict: int = 256,
    ) -> str:
        msg: dict = {"role": "user", "content": prompt}
        if b64s:
            msg["images"] = b64s
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
            "messages": [msg],
        }
        if json_mode:
            body["format"] = "json"
        if self._send_think:
            body["think"] = False  # disable chain-of-thought (faster; Gemma 4 etc.)
        resp = self._requests.post(f"{self.host}/api/chat", json=body, timeout=120)
        if resp.status_code != 200:
            txt = resp.text.strip()
            # Older/non-thinking models reject the think flag — drop it and retry once.
            if self._send_think and "think" in txt.lower():
                self._send_think = False
                return self._chat(b64s, prompt, temperature, json_mode, num_predict)
            raise RuntimeError(
                f"Ollama {resp.status_code}: {txt[:200]} "
                f"— is the model pulled?  run: ollama pull {self.model}"
            )
        return resp.json()["message"]["content"] or ""

    def check(self, jpeg: bytes, rule: str) -> Verdict:
        b64s = [base64.b64encode(jpeg).decode()]
        prompt = build_prompt(rule, self.sensitivity)
        content = self._chat(b64s, prompt, temperature=0.0)
        if not content.strip():
            # Greedy JSON decoding collapses to an empty string on some small
            # models for certain prompts; a little temperature breaks the tie
            # while keeping the output short and JSON-constrained.
            content = self._chat(b64s, prompt, temperature=0.5)
        return _parse_verdict(content)

    def generate(self, prompt: str, images: bytes | list[bytes] | None = None, json_mode: bool = True) -> str:
        b64s = [base64.b64encode(j).decode() for j in self._as_list(images)]
        # World-state updates are bigger than a verdict; give them headroom.
        return self._chat(b64s, prompt, temperature=0.0, json_mode=json_mode, num_predict=1024)


class GeminiBackend(VLMBackend):
    """Cloud VLM via Google Gemini. Needs GEMINI_API_KEY."""

    name = "gemini"
    box_format = "yxyx"  # Gemini's grounding is trained on box_2d = [ymin,xmin,ymax,xmax]

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None, sensitivity: str = "strict"):
        from google import genai
        from google.genai import types

        self._types = types
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Set GEMINI_API_KEY to use the Gemini backend "
                "(export GEMINI_API_KEY=...)."
            )
        self._client = genai.Client(api_key=key)
        self.model = model
        self.sensitivity = sensitivity
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._usage_lock = threading.Lock()

    def _call(self, prompt: str, images: list[bytes], json_mode: bool) -> str:
        cfg = {}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        # "thinking" exists only on Gemini 2.5 Flash; disabling it cuts latency.
        # Gemma / Pro reject thinking_config, so set it only for gemini flash.
        if self.model.startswith("gemini") and "pro" not in self.model:
            cfg["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
        contents: list = [
            self._types.Part.from_bytes(data=j, mime_type="image/jpeg") for j in images
        ]
        contents.append(prompt)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(**cfg),
        )
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            with self._usage_lock:
                self.usage["calls"] += 1
                self.usage["input_tokens"] += int(getattr(um, "prompt_token_count", 0) or 0)
                self.usage["output_tokens"] += int(getattr(um, "candidates_token_count", 0) or 0)
        return _gemini_text(resp)

    # $/M tokens for gemini-3.1-flash-lite (2026-07); other models differ, so
    # the cost line is approximate — token counts are always exact.
    PRICE_IN, PRICE_OUT = 0.25, 1.50

    def usage_cost(self, usage: dict | None = None) -> float:
        u = usage or self.usage
        return u["input_tokens"] / 1e6 * self.PRICE_IN + u["output_tokens"] / 1e6 * self.PRICE_OUT

    def usage_line(self, usage: dict | None = None) -> str:
        u = usage or self.usage
        return (
            f"{u['calls']} calls · {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens"
            f" · ~${self.usage_cost(u):.4f}"
        )

    def check(self, jpeg: bytes, rule: str) -> Verdict:
        return _parse_verdict(self._call(build_prompt(rule, self.sensitivity), [jpeg], True))

    def generate(self, prompt: str, images: bytes | list[bytes] | None = None, json_mode: bool = True) -> str:
        return self._call(prompt, self._as_list(images), json_mode)


def make_backend(
    name: str, model: str | None = None, sensitivity: str | None = None
) -> VLMBackend:
    name = name.lower()
    if name == "ollama":
        # Local model is the cheap first-pass filter → default to high-recall.
        return OllamaBackend(model=model or "qwen2.5vl", sensitivity=sensitivity or "liberal")
    if name == "gemini":
        return GeminiBackend(model=model or "gemini-2.5-flash", sensitivity=sensitivity or "strict")
    raise ValueError(f"unknown backend: {name!r} (choose 'ollama' or 'gemini')")
