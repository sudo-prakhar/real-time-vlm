# GAVI web app — persistent container (Railway/Fly/Render).
FROM python:3.11-slim

# opencv-python-headless still wants these at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Headless server deps: opencv-python-headless instead of opencv-python
# (no GUI stack in a container; cv2.imshow is only used by the local CLI).
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] opencv-python-headless numpy scipy \
    requests google-genai

COPY gavi/ gavi/
COPY web/ web/

# Railway injects PORT; default for local docker runs
ENV PORT=8000
CMD ["sh", "-c", "uvicorn gavi.server:app --host 0.0.0.0 --port ${PORT}"]
