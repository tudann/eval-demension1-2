from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "co-tracker"


@lru_cache(maxsize=1)
def _model(device: str):
    import torch
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from cotracker.predictor import CoTrackerPredictor
    checkpoint = ROOT / "models" / "cotracker" / "scaled_offline.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return CoTrackerPredictor(checkpoint=str(checkpoint), offline=True).to(device).eval()


def infer(frames: list[np.ndarray], width: int, height: int, device: str = "cuda", grid_size: int = 12) -> dict:
    import torch
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    video = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float()[None].to(device)
    with torch.inference_mode():
        tracks, visibility = _model(device)(video, grid_size=grid_size, backward_tracking=True)
    tracks = tracks[0].float().cpu().numpy()
    visibility = visibility[0].bool().cpu().numpy()
    diagonal = float(np.hypot(width, height))
    velocity = np.diff(tracks, axis=0) / diagonal
    speed = np.linalg.norm(velocity, axis=2)
    pair_visible = visibility[:-1] & visibility[1:]
    valid_speed = speed[pair_visible]
    acceleration = np.diff(velocity, axis=0)
    triple_visible = visibility[:-2] & visibility[1:-1] & visibility[2:]
    valid_acceleration = np.linalg.norm(acceleration, axis=2)[triple_visible]
    jumps = valid_speed > max(.05, float(np.median(valid_speed) * 5)) if valid_speed.size else np.array([])
    return {
        "visibility_rate": float(visibility.mean()),
        "normalized_speed_mean": float(valid_speed.mean()) if valid_speed.size else None,
        "normalized_speed_std": float(valid_speed.std()) if valid_speed.size else None,
        "normalized_acceleration_mean": float(valid_acceleration.mean()) if valid_acceleration.size else None,
        "jump_rate": float(jumps.mean()) if jumps.size else None,
        "track_count": int(tracks.shape[1]), "frame_count": int(tracks.shape[0]),
        "grid_size": grid_size, "source": "CoTracker3 scaled_offline",
        "checkpoint": str(ROOT / "models/cotracker/scaled_offline.pth"),
    }
