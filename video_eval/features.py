from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class VideoFeatures:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    sample_indices: list[int]
    frames: list[np.ndarray]
    gray: list[np.ndarray]
    timestamps: list[float]

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


def decode(path: str | Path, sample_fps: float = 4.0, max_frames: int = 96) -> VideoFeatures:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot decode video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("invalid video metadata")
    stride = max(1, round(fps / sample_fps))
    candidates = list(range(0, count, stride))
    if len(candidates) > max_frames:
        candidates = np.linspace(0, count - 1, max_frames, dtype=int).tolist()
    frames, indices = [], []
    wanted = iter(candidates)
    target = next(wanted, None)
    index = 0
    while target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        if index == target:
            frames.append(frame)
            indices.append(index)
            target = next(wanted, None)
        index += 1
    cap.release()
    if len(frames) < 2:
        raise ValueError("video yielded fewer than two sampled frames")
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    return VideoFeatures(path, fps, count, width, height, indices, frames, gray, [i / fps for i in indices])


def objective_signals(video: VideoFeatures) -> dict[str, float | list[float]]:
    sharpness = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in video.gray]
    brightness = [float(g.mean()) for g in video.gray]
    contrast = [float(g.std()) for g in video.gray]
    entropy = []
    for gray in video.gray:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        prob = hist / max(float(hist.sum()), 1.0)
        prob = prob[prob > 0]
        entropy.append(float(-(prob * np.log2(prob)).sum()))
    flows, residuals = [], []
    for before, after in zip(video.gray, video.gray[1:]):
        scale = min(1.0, 640.0 / max(before.shape))
        size = (max(8, round(before.shape[1] * scale)), max(8, round(before.shape[0] * scale)))
        a, b = cv2.resize(before, size), cv2.resize(after, size)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=2)
        flows.append(float(np.median(magnitude)))
        residuals.append(float(np.mean(cv2.absdiff(a, b))))
    return {
        "sharpness_mean": float(np.mean(sharpness)), "sharpness_p10": float(np.percentile(sharpness, 10)),
        "brightness_mean": float(np.mean(brightness)), "brightness_std": float(np.std(brightness)),
        "contrast_mean": float(np.mean(contrast)), "entropy_mean": float(np.mean(entropy)),
        "flow_mean": float(np.mean(flows)), "flow_std": float(np.std(flows)),
        "flow_acceleration": float(np.mean(np.abs(np.diff(flows)))) if len(flows) > 1 else 0.0,
        "frame_residual_mean": float(np.mean(residuals)), "frame_residual_std": float(np.std(residuals)),
        "sharpness_series": sharpness, "brightness_series": brightness, "flow_series": flows,
    }
