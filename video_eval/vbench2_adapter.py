from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "VBench" / "VBench-2.0"
MODELS = ROOT / "models" / "vbench2"
ISOLATED_PYTHON = ROOT / "envs" / "vbench2" / "bin" / "python"
WORKER = ROOT / "video_eval" / "vbench2_worker.py"


def _isolated_human_anatomy(video_path: str) -> dict[str, Any]:
    """Run the OpenMMLab stack in its Torch 2.1/MMCV 2.1 environment."""
    env = os.environ.copy()
    yolo_world = REPO / "vbench2" / "third_party" / "YOLO-World"
    env["PYTHONPATH"] = os.pathsep.join((
        str(ROOT), str(REPO), str(yolo_world), str(yolo_world / "mmyolo")
    ))
    env.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
    env["VBENCH2_ISOLATED_WORKER"] = "1"
    if env.get("VIDEO_EVAL_ANATOMY_DEVICE") == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(
        [str(ISOLATED_PYTHON), str(WORKER), "human_anatomy", video_path],
        env=env, text=True, capture_output=True, timeout=1800,
    )
    marker = "__VBENCH2_RESULT__="
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    detail = (process.stderr or process.stdout)[-4000:]
    raise RuntimeError(
        f"isolated VBench2 worker exited {process.returncode} without a result: {detail}"
    )


def _ensure_torchvision_write_video() -> None:
    """Compatibility for torchvision releases that removed io.write_video.

    VBench-2.0 imports the symbol from its shared utils even though the four
    selected metrics do not normally call it.  The shim preserves the expected
    THWC uint8 contract and keeps repository imports functional.
    """
    import cv2
    import numpy as np
    import torchvision.io
    if hasattr(torchvision.io, "write_video"):
        return

    def write_video(filename, video_array, fps, video_codec="h264", options=None, **_kwargs):
        frames = video_array.detach().cpu().numpy() if hasattr(video_array, "detach") else np.asarray(video_array)
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError("write_video compatibility shim expects THWC RGB/RGBA frames")
        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {filename}")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    torchvision.io.write_video = write_video


def _ensure_timm_legacy_api() -> None:
    """Expose the renamed interpolation helper expected by ViTDetector."""
    import timm.data.transforms as transforms
    if not hasattr(transforms, "_pil_interp"):
        transforms._pil_interp = transforms.str_to_pil_interp


@contextmanager
def _repo_context():
    """VBench-2.0 uses repository-relative configs, so isolate its cwd change."""
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        _ensure_torchvision_write_video()
        _ensure_timm_legacy_api()
        os.chdir(REPO)
        yolo_world = REPO / "vbench2" / "third_party" / "YOLO-World"
        sys.path.insert(0, str(yolo_world / "mmyolo"))
        sys.path.insert(0, str(yolo_world))
        sys.path.insert(0, str(REPO / "vbench2" / "third_party" / "LLaVA_NeXT"))
        sys.path.insert(0, str(REPO))
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _metadata(video_path: str, dimension: str, prompt: str = "") -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump([{"dimension": [dimension], "prompt_en": prompt,
                "video_list": [str(Path(video_path).resolve())]}], handle)
    handle.close()
    return handle.name


def _official_call(video_path: str, dimension: str, function: Callable[..., Any],
                   submodules: dict[str, Any], prompt: str = "", device: str = "cuda") -> dict[str, Any]:
    metadata = _metadata(video_path, dimension, prompt)
    try:
        value, per_video = function(metadata, device, submodules)
    finally:
        Path(metadata).unlink(missing_ok=True)
    item = per_video[0] if per_video else {}
    raw = item.get("video_results", value)
    if raw == -1:
        return {"applicable": False, "value": None, "official_output": item,
                "reason": "VBench-2.0 marked this video invalid/not applicable."}
    return {"applicable": True, "value": float(raw), "aggregate": float(value),
            "official_output": item, "source": f"VBench-2.0 {dimension}", "execution_device": device}


ANATOMY_SAMPLE_FPS = 2
ANATOMY_MAX_FRAMES = 48
ANATOMY_THRESHOLDS = {
    "human": 0.4545454545454546,
    "face": 0.30303030303030304,
    "hand": 0.3232,
}
ANATOMY_WEIGHTS = {
    "human": MODELS / "anomaly_detector" / "human.pth",
    "face": MODELS / "anomaly_detector" / "face.pth",
    "hand": MODELS / "anomaly_detector" / "hand.pth",
}


def _sample_bgr_frames(video_path: str) -> list[Any]:
    """Use the same 2 fps / 48 frame budget as the evaluation request."""
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        capture.release()
        raise RuntimeError(f"video has no readable frames: {video_path}")
    stride = max(1, round(fps / ANATOMY_SAMPLE_FPS))
    indices = list(range(0, count, stride))
    if len(indices) > ANATOMY_MAX_FRAMES:
        indices = np.linspace(0, count - 1, ANATOMY_MAX_FRAMES, dtype=int).tolist()
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video yielded no sampled frames: {video_path}")
    return frames


@lru_cache(maxsize=1)
def _grounding_detector(device: str):
    from .grounded_sam2_adapter import REPO, _ensure_repo_path
    _ensure_repo_path()
    from grounding_dino.groundingdino.util.inference import load_model
    return load_model(
        str(REPO / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        str(ROOT / "models/groundingdino/groundingdino_swint_ogc.pth"), device=device)


def _detect_boxes(model, frame, caption: str, allowed: tuple[str, ...], device: str) -> list[tuple[Any, str]]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.ops import box_convert, nms
    from grounding_dino.groundingdino.datasets import transforms as T
    from grounding_dino.groundingdino.util.inference import predict
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor, _ = transform(Image.fromarray(rgb), None)
    boxes, confidence, names = predict(model, tensor, caption, 0.25, 0.25, device=device)
    if len(boxes) == 0:
        return []
    height, width = rgb.shape[:2]
    xyxy = box_convert(boxes * torch.tensor([width, height, width, height], device=boxes.device), "cxcywh", "xyxy")
    keep = nms(xyxy, confidence, 0.5).tolist()
    found = []
    for index in keep:
        label = next((item for item in allowed if item in names[index].lower()), None)
        if label is None:
            continue
        box = xyxy[index].detach().cpu().numpy()
        box[0::2] = np.clip(box[0::2], 0, width - 1)
        box[1::2] = np.clip(box[1::2], 0, height - 1)
        if box[2] > box[0] and box[3] > box[1]:
            found.append((box, label))
    return found


def _anatomy_detections(frames: list[Any], device: str) -> list[dict[str, Any]]:
    model = _grounding_detector(device)
    rows = []
    for frame_index, frame in enumerate(frames):
        humans = _detect_boxes(model, frame, "human.", ("human",), device)
        for person_index, (box, _label) in enumerate(humans):
            rows.append({"frame_index": frame_index, "person_index": person_index,
                         "bbox": box.tolist(), "label": "human"})
            x1, y1, x2, y2 = (int(round(value)) for value in box.tolist())
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue
            for part_box, label in _detect_boxes(model, crop, "face. hand.", ("face", "hand"), device):
                shifted = part_box.copy()
                shifted[0::2] += x1
                shifted[1::2] += y1
                rows.append({"frame_index": frame_index, "person_index": person_index,
                             "bbox": shifted.tolist(), "label": label})
    return rows


def _square_crop(frame, bbox) -> Any | None:
    """Keep the VBench ViT crop: a square around the box, then the inner fallback."""
    x1, y1, x2, y2 = map(float, bbox)
    height, width = frame.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

    def cut(side: float, outer: bool):
        if outer:
            x_min = max(x2 - side / 2, side / 2)
            x_max = min(x1 + side / 2, width - side / 2)
            y_min = max(y2 - side / 2, side / 2)
            y_max = min(y1 + side / 2, height - side / 2)
        else:
            x_min, x_max = max(side / 2, 0.0), width - side / 2
            y_min, y_max = max(side / 2, 0.0), height - side / 2
        if x_min > x_max or y_min > y_max:
            return None
        center_x = min(max(mid_x, x_min), x_max)
        center_y = min(max(mid_y, y_min), y_max)
        origin_x, origin_y = center_x - side / 2, center_y - side / 2
        left, top = math.floor(origin_x), math.floor(origin_y)
        right, bottom = math.ceil(origin_x + side), math.ceil(origin_y + side)
        if left < 0 or top < 0 or right > width or bottom > height:
            return None
        cropped = frame[top:bottom, left:right]
        return cropped if cropped.size else None

    cropped = cut(max(box_w, box_h), True)
    if cropped is None:
        cropped = cut(min(box_w, box_h), False)
    return cropped


@lru_cache(maxsize=1)
def _anatomy_models(device: str):
    import torch
    from torchvision import transforms
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    _ensure_timm_legacy_api()
    _ensure_torchvision_write_video()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from vbench2.third_party.ViTDetector.config import get_config
    from vbench2.third_party.ViTDetector.models import build_model
    from timm.data.transforms import _pil_interp
    config_path = REPO / "vbench2/third_party/ViTDetector/simmim_finetune__vit_base__img224__800ep.yaml"
    args = type("Args", (), {"cfg": str(config_path), "opts": None, "local_rank": 0})()
    models, image_transforms = {}, {}
    for name, weight in ANATOMY_WEIGHTS.items():
        if not weight.is_file():
            raise FileNotFoundError(weight)
        config = get_config(args)
        model = build_model(config, is_pretrain=False)
        checkpoint = torch.load(weight, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        models[name] = model.to(device).eval()
        size = int((256 / 224) * config.DATA.IMG_SIZE)
        image_transforms[name] = transforms.Compose([
            transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
            transforms.CenterCrop(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    return models, image_transforms


def _score_anatomy(frames: list[Any], detections: list[dict[str, Any]], device: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    models, image_transforms = _anatomy_models(device)
    by_frame = defaultdict(list)
    for item in detections:
        by_frame[item["frame_index"]].append(item)
    abnormal_people = 0
    total_people = 0
    for frame_index, frame in enumerate(frames):
        people = defaultdict(lambda: defaultdict(list))
        for item in by_frame.get(frame_index, []):
            people[item["person_index"]][item["label"]].append(item["bbox"])
        total_people += len(people)
        predictions = defaultdict(lambda: defaultdict(list))
        for person_index, categories in people.items():
            for category, boxes in categories.items():
                crops = [crop for box in boxes if (crop := _square_crop(frame, box)) is not None]
                if not crops:
                    continue
                batch = torch.cat([image_transforms[category](Image.fromarray(crop)).unsqueeze(0) for crop in crops]).to(device)
                with torch.no_grad():
                    scores = F.softmax(models[category](batch), dim=1).detach().cpu().tolist()
                predictions[person_index][category].extend(scores)
        for person_index in people:
            abnormal = any(score[0] > ANATOMY_THRESHOLDS[category]
                           for category, scores in predictions[person_index].items() for score in scores)
            abnormal_people += int(abnormal)
    ratio = abnormal_people / total_people if total_people else 0.0
    return {"video_results": 1 - ratio, "person_count": total_people, "abnormal_count": abnormal_people}


def human_anatomy(video_path: str) -> dict[str, Any]:
    """Score anatomy with the original ViT detectors and this tree's GroundingDINO boxes.

    YOLO-World stays out of this path because its MMCV CUDA kernels do not run
    on this machine. The abnormality thresholds and 1 - abnormal_rate formula
    stay the same. Frames follow the evaluation budget of 2 fps and 48 frames.
    """
    device = "cpu" if os.getenv("VIDEO_EVAL_ANATOMY_DEVICE") == "cpu" else "cuda"
    frames = _sample_bgr_frames(video_path)
    detections = _anatomy_detections(frames, device)
    measured = _score_anatomy(frames, detections, device)
    value = float(measured["video_results"])
    return {"applicable": True, "value": value, "aggregate": value,
            "official_output": {"video_path": str(Path(video_path).resolve()), "video_results": value},
            "source": "VBench-2.0 human anatomy ViT; detector GroundingDINO Swin-T",
            "execution_device": device, "detector": "GroundingDINO Swin-T",
            "sampled_frames": len(frames), "person_count": measured["person_count"],
            "abnormal_count": measured["abnormal_count"]}


def human_identity(video_path: str) -> dict[str, Any]:
    weight = MODELS / "arcface" / "resnet18_110.pth"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    with _repo_context():
        import decord
        with decord.bridge.use_torch():
            from vbench2.human_identity import compute_human_identity
            return _official_call(video_path, "human_identity", compute_human_identity, {"model": str(weight)})


def human_clothes(video_path: str) -> dict[str, Any]:
    """Run official VBench-2.0 Human Clothes with its published LLaVA-Video model.

    This backend is an explicit user-authorized exception to plan.md's original
    single-general-VLM restriction and is therefore never used by another metric.
    """
    model = ROOT / "models" / "lmms-lab" / "LLaVA-Video-7B-Qwen2"
    required = model / "model.safetensors.index.json"
    if not required.is_file():
        raise FileNotFoundError(required)
    with _repo_context():
        from vbench2.human_clothes import compute_human_clothes
        return _official_call(video_path, "human_clothes", compute_human_clothes,
                              {"llava": str(model)})


def multi_view_consistency(video_path: str) -> dict[str, Any]:
    """Official VBench-2.0 multi-view path (CoTracker camera orbit + RAFT + dense match)."""
    # Release cached detection weights and unreachable model/offload-hook cycles
    # before the full-video tracker. Evidence already stored is plain CPU data.
    import gc
    import torch
    from video_eval import grounded_sam2_adapter
    grounded_sam2_adapter._models.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    raft = ROOT / "models" / "raft" / "raft-things.pth"
    if not raft.is_file():
        raise FileNotFoundError(raft)
    # The official code obtains CoTracker through torch.hub. Its repository/model
    # names are retained here for traceability; runtime download is never silently replaced.
    submodules = {"raft": str(raft), "repo": "facebookresearch/co-tracker:main", "model": "cotracker2"}
    with _repo_context():
        import decord
        with decord.bridge.use_torch():
            from vbench2.multi_view_consistency import compute_multi_view_consistency
            return _official_call(video_path, "multi-view_consistency", compute_multi_view_consistency, submodules)

