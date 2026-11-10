from pathlib import Path

import cv2
import numpy as np

path = Path("/tmp/video_eval_smoke.mp4")
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (320, 240))
assert writer.isOpened()
for index in range(24):
    frame = np.full((240, 320, 3), (35, 70, 120), dtype=np.uint8)
    cv2.circle(frame, (50 + index * 6, 120), 24, (20, 220, 240), -1)
    writer.write(frame)
writer.release()
print(path)
