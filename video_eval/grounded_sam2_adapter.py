from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Grounded-SAM-2"


def _ensure_repo_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


@lru_cache(maxsize=1)
def _models(device: str):
    _ensure_repo_path()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml",
                     str(ROOT / "models/sam2/sam2.1_hiera_small.pt"), device=device)
    grounding = load_model(
        str(REPO / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        str(ROOT / "models/groundingdino/groundingdino_swint_ogc.pth"), device=device)
    return SAM2ImagePredictor(sam), grounding


def infer(frames: list[np.ndarray], timestamps: list[float], labels: list[str], device: str = "cuda",
          box_threshold: float = .35, text_threshold: float = .25) -> dict:
    _ensure_repo_path()
    import torch
    from torchvision.ops import box_convert, nms
    from grounding_dino.groundingdino.datasets import transforms as T
    from grounding_dino.groundingdino.util.inference import predict
    predictor, grounding = _models(device)
    clean = list(dict.fromkeys(label.strip().lower().rstrip(".") for label in labels if label.strip()))
    if not clean:
        raise ValueError("Grounded-SAM-2 requires at least one text label")
    caption = ". ".join(clean) + "."
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    rows = []
    for bgr, timestamp in zip(frames, timestamps):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, _ = transform(Image.fromarray(rgb), None)
        boxes, confidence, names = predict(grounding, tensor, caption, box_threshold,
                                           text_threshold, device=device)
        h, w = rgb.shape[:2]
        xyxy = box_convert(boxes * torch.tensor([w, h, w, h]), "cxcywh", "xyxy").cpu()
        keep = nms(xyxy, confidence.cpu(), .5).tolist() if len(xyxy) else []
        xyxy = xyxy[keep].numpy() if keep else np.empty((0, 4), dtype=np.float32)
        confidence = confidence[keep].cpu().numpy() if keep else np.empty(0)
        names = [names[index] for index in keep]
        masks, mask_scores = [], []
        if len(xyxy):
            predictor.set_image(rgb)
            raw_masks, raw_scores, _ = predictor.predict(box=xyxy, multimask_output=False)
            masks = raw_masks.squeeze(1).astype(bool) if raw_masks.ndim == 4 else raw_masks.astype(bool)
            mask_scores = np.asarray(raw_scores).reshape(-1)
        annotations = []
        for index, (name, box, score) in enumerate(zip(names, xyxy, confidence)):
            mask = masks[index]
            ys, xs = np.where(mask)
            annotations.append({"label": name, "box_xyxy": box.tolist(), "box_confidence": float(score),
                                "mask_score": float(mask_scores[index]), "mask_area_ratio": float(mask.mean()),
                                "mask_centroid": [float(xs.mean()), float(ys.mean())] if len(xs) else None})
        rows.append({"timestamp": timestamp, "annotations": annotations})
    return {"query_labels": clean, "frames": rows, "source": "GroundingDINO Swin-T + SAM 2.1 Hiera Small",
            "grounding_checkpoint": str(ROOT / "models/groundingdino/groundingdino_swint_ogc.pth"),
            "sam_checkpoint": str(ROOT / "models/sam2/sam2.1_hiera_small.pt")}


def summary(result: dict) -> dict:
    labels = result["query_labels"]
    counts = {label: [] for label in labels}
    for row in result["frames"]:
        for label in labels:
            counts[label].append(sum(label in item["label"].lower() for item in row["annotations"]))
    return {
        "per_label": {label: {"counts": values, "presence_rate": float(np.mean(np.array(values) > 0)),
                              "median_count": float(np.median(values)), "count_std": float(np.std(values))}
                      for label, values in counts.items()},
        "frames": result["frames"], "source": result["source"],
        "grounding_checkpoint": result["grounding_checkpoint"], "sam_checkpoint": result["sam_checkpoint"],
    }
