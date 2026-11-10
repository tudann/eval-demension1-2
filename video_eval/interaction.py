from __future__ import annotations

from typing import Any

import numpy as np


HAND_IDS = (9, 10)  # COCO left/right wrist; DWPose also includes whole-body hands.
FOOT_IDS = (15, 16)
MOUTH_IDS = (0,)     # nose is used only as a coarse mouth/head-contact proxy.


def _point_box_distance(point: np.ndarray, box: list[float], diagonal: float) -> float:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy) / max(diagonal, 1.0))


def contact_signals(pose_result: dict[str, Any], grounding: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Align pose and object detections by sampled frame and measure 2-D proximity.

    This deliberately reports *candidate* contact only; 2-D overlap is not called
    physical contact, matching plan.md's limitation.
    """
    diagonal = float(np.hypot(width, height))
    rows = []
    for pose_row, det_row in zip(pose_result["frames"], grounding["frames"]):
        objects = [a for a in det_row["annotations"] if "person" not in a["label"].lower()]
        candidates = []
        for person_index, (points, scores) in enumerate(zip(pose_row["keypoints"], pose_row["scores"])):
            for part, indices in (("hand", HAND_IDS), ("foot", FOOT_IDS), ("mouth_proxy", MOUTH_IDS)):
                for keypoint_index in indices:
                    if keypoint_index >= len(points) or scores[keypoint_index] <= .3:
                        continue
                    for obj in objects:
                        distance = _point_box_distance(points[keypoint_index], obj["box_xyxy"], diagonal)
                        candidates.append({"person": person_index, "part": part, "keypoint": keypoint_index,
                                           "object": obj["label"], "distance_norm": distance,
                                           "candidate_contact": distance <= .025})
        rows.append({"timestamp": det_row["timestamp"], "candidates": candidates})
    flat = [c for row in rows for c in row["candidates"]]
    contact = [c for c in flat if c["candidate_contact"]]
    return {"frames": rows, "candidate_contact_rate": len(contact) / len(flat) if flat else None,
            "candidate_contact_frames": sum(any(c["candidate_contact"] for c in row["candidates"]) for row in rows),
            "minimum_distance_norm": min((c["distance_norm"] for c in flat), default=None),
            "limitation": "2-D keypoint/box proximity is only candidate contact, not proof of physical contact."}


def object_motion(grounding: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    diagonal = float(np.hypot(width, height))
    labels: dict[str, list[tuple[float, np.ndarray]]] = {}
    for row in grounding["frames"]:
        for item in row["annotations"]:
            if "person" in item["label"].lower() or item.get("mask_centroid") is None:
                continue
            labels.setdefault(item["label"], []).append((row["timestamp"], np.asarray(item["mask_centroid"], float)))
    summaries = {}
    for label, track in labels.items():
        displacements = [float(np.linalg.norm(b[1] - a[1]) / diagonal) for a, b in zip(track, track[1:])]
        summaries[label] = {"observations": len(track), "timestamps": [x[0] for x in track],
                            "step_displacements_norm": displacements,
                            "total_displacement_norm": float(np.linalg.norm(track[-1][1] - track[0][1]) / diagonal) if len(track) > 1 else 0.0,
                            "tail_stability": float(np.mean(displacements[-max(1, len(displacements)//3):])) if displacements else None}
    return {"objects": summaries, "source": "Grounded-SAM-2 mask-centroid temporal association by label"}


def interaction_applicable(prompt_info: dict[str, Any] | None, grounding: dict[str, Any] | None) -> bool:
    """Decide whether interaction metrics should run.

    A parsed prompt is enough. The parser's has_interaction flag is not a veto:
    it marked hand-picked interaction clips as false, and a person-only detector
    then found no object, so every D6 metric was discarded. Individual metrics
    still return insufficient evidence when contact or object motion is missing.
    """
    if prompt_info is not None:
        return True
    if grounding:
        return any("person" not in a["label"].lower() for row in grounding["frames"] for a in row["annotations"])
    return False
