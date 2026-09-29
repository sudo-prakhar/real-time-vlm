# Convenience runners.  Override the rule:  make online RULE="the door is left open"
RULE ?= a person is raising their hand
OFFLINE_MODEL ?= gemma4:26b
MAXSIDE ?= 512

# Cloud — low latency + high accuracy. Concurrency helps (calls are I/O-bound),
# so we run a pool of workers.
online:
	python -m gavi monitor --backend gemini --model gemini-3.1-flash-lite --workers 6 --interval 0 \
		--rule "$(RULE)" --display

# Local / fully offline — Ollama + Gemma. Local inference is COMPUTE-bound, so 1
# worker is correct; more just thrash the CPU.
# First:  ollama pull $(OFFLINE_MODEL)   ·   compare: make offline OFFLINE_MODEL=gemma3:4b
offline:
	python -m gavi monitor --backend ollama --model $(OFFLINE_MODEL) --workers 1 --interval 0 \
		--max-side $(MAXSIDE) --rule "$(RULE)" --display

# World model (Phase 2) — persistent entity tracking + timeline + Q&A.
# Rule is optional here:  make world   ·   make world RULE="the desk is unattended"
# While running, type `? <question>` to query the world. Resumes world/ by default.
WORLD_RULE = $(if $(filter command line,$(origin RULE)),--rule "$(RULE)")
world:
	python -m gavi world --backend gemini --model gemini-3.1-flash-lite --interval 0 \
		$(WORLD_RULE) --display

world-offline:
	python -m gavi world --backend ollama --model $(OFFLINE_MODEL) --interval 0 \
		--max-side $(MAXSIDE) $(WORLD_RULE) --display

# GAVI web app — landing page + live browser demo at http://localhost:8000
web:
	python -m uvicorn gavi.server:app --host 0.0.0.0 --port 8000

# Rebuild web/assets/hero.mp4 + hero_tracks.json (needs ffmpeg + GEMINI_API_KEY)
hero:
	python scripts/make_hero.py

# Which webcam index works? (diagnoses black frames / missing Camera permission)
camtest:
	python scripts/camtest.py

.PHONY: online offline world world-offline web hero camtest
