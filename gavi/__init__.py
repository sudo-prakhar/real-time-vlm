"""GAVI — real-time video understanding with vision-language models.

Modules:
  backends       pluggable VLM backends (Ollama local, Gemini cloud)
  world          persistent world model: entity registry + event timeline
  identity       deterministic appearance + motion identity matcher
  engine         the shared OBSERVE -> MATCH -> APPLY -> REASON cycle
  video, utils   frame/source helpers, .env loading, alerts

Entry points:
  monitor        phase-1 stateless rule monitor   (python -m gavi monitor)
  world_monitor  phase-2 world-model monitor      (python -m gavi world)
  server         FastAPI web app                  (uvicorn gavi.server:app)
"""
