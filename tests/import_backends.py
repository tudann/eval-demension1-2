import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "third_party/Grounded-SAM-2", ROOT / "third_party/co-tracker",
             ROOT / "third_party/Video-Depth-Anything", ROOT / "third_party/DWPose/ControlNet-v1-1-nightly"):
    sys.path.insert(0, str(path))

from sam2.build_sam import build_sam2  # noqa: F401
from cotracker.predictor import CoTrackerPredictor  # noqa: F401
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: F401

print("backend-imports-ok")
