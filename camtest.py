#!/usr/bin/env python3
"""Diagnose which camera index works. Run:  python camtest.py

Reports each index: can it open, and are the frames real (not black)?
A black result on macOS means the app running Python lacks Camera permission.
"""

import time

import cv2

for idx in range(4):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"index {idx}: cannot open")
        cap.release()
        continue
    means = []
    for _ in range(10):  # warm-up + sample
        ok, frame = cap.read()
        if ok and frame is not None:
            means.append(float(frame.mean()))
        time.sleep(0.05)
    cap.release()
    if not means:
        print(f"index {idx}: opened but returned no frames")
        continue
    avg = sum(means) / len(means)
    verdict = "BLACK — likely Camera permission" if avg < 5 else "OK — real frames"
    print(f"index {idx}: opened, avg brightness {avg:5.1f}  ->  {verdict}")

print("\nUse a working index with:  python run.py --source <idx> --rule '...' --display")
