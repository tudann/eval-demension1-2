from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Video-Depth-Anything"

@contextmanager
def _vendor_utils():
    """Scope the vendor's generic 'utils' import without poisoning other backends."""
    previous={k:v for k,v in sys.modules.items() if k=="utils" or k.startswith("utils.")}
    old_path=list(sys.path)
    for key in previous:
        sys.modules.pop(key,None)
    sys.path.insert(0,str(REPO))
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key=="utils" or key.startswith("utils."):
                sys.modules.pop(key,None)
        sys.modules.update(previous)
        sys.path[:]=old_path


@lru_cache(maxsize=1)
def _model(device: str):
    import torch
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    with _vendor_utils():
        from video_depth_anything.video_depth import VideoDepthAnything
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    checkpoint = ROOT / "models" / "video_depth_anything" / "video_depth_anything_vits.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def infer(frames: list[np.ndarray], fps: float, device: str = "cuda") -> dict:
    array = np.stack([frame[..., ::-1] for frame in frames])  # BGR -> RGB
    depths, output_fps = _model(device).infer_video_depth(array, fps, input_size=518, device=device, fp32=False)
    medians = np.median(depths, axis=(1, 2))
    return {
        "depths": depths,
        "frame_medians": medians.tolist(),
        "temporal_median_delta": float(np.mean(np.abs(np.diff(medians)))) if len(medians) > 1 else 0.0,
        "output_fps": float(output_fps), "source": "Video Depth Anything Small",
        "checkpoint": str(ROOT / "models/video_depth_anything/video_depth_anything_vits.pth"),
    }
