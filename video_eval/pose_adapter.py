from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import os

ROOT = Path(__file__).resolve().parents[1]
DWPOSE = ROOT / "third_party" / "DWPose" / "ControlNet-v1-1-nightly"


@lru_cache(maxsize=1)
def _sessions():
    import onnxruntime as ort
    if str(DWPOSE) not in sys.path:
        sys.path.insert(0, str(DWPOSE))
    detector = ROOT / "models" / "dwpose" / "yolox_l.onnx"
    pose = ROOT / "models" / "dwpose" / "dw-ll_ucoco_384.onnx"
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available and os.getenv("VIDEO_EVAL_ONNX_DEVICE", "cuda") != "cpu" else ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return (ort.InferenceSession(str(detector), sess_options=options, providers=providers),
            ort.InferenceSession(str(pose), sess_options=options, providers=providers), providers)


def infer(frames: list[np.ndarray], timestamps: list[float]) -> dict:
    """Run official DWPose whole-body ONNX detector/pose models."""
    if str(DWPOSE) not in sys.path:
        sys.path.insert(0, str(DWPOSE))
    from annotator.dwpose.onnxdet import inference_detector
    from annotator.dwpose.onnxpose import inference_pose
    detector, pose, providers = _sessions()
    rows = []
    for frame, timestamp in zip(frames, timestamps):
        detections = inference_detector(detector, frame)
        keypoints, scores = inference_pose(pose, detections, frame)
        rows.append({"timestamp": timestamp, "keypoints": keypoints, "scores": scores})
    return {"frames": rows, "providers": providers,
            "detector_weight": str(ROOT / "models/dwpose/yolox_l.onnx"),
            "pose_weight": str(ROOT / "models/dwpose/dw-ll_ucoco_384.onnx"),
            "source": "DWPose Wholebody (RTMPose whole-body ONNX)"}


def signals(result: dict, width: int, height: int) -> dict:
    rows = result["frames"]
    people = [len(row["keypoints"]) for row in rows]
    visibility = []
    centers = []
    body_ratios = []
    angle_rows = []
    for row in rows:
        if not len(row["keypoints"]):
            visibility.append(0.0)
            centers.append(None)
            continue
        scores = row["scores"][0]
        points = row["keypoints"][0]
        visible = scores > .3
        visibility.append(float(visible.mean()))
        centers.append(points[visible].mean(axis=0) if visible.any() else None)
        # COCO body: shoulders 5/6, hips 11/12, knees 13/14, ankles 15/16.
        lengths = []
        for a, b in ((5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
            if a < len(points) and b < len(points) and scores[a] > .3 and scores[b] > .3:
                lengths.append(float(np.linalg.norm(points[a] - points[b]) / max(width, height)))
        body_ratios.append(float(np.mean(lengths)) if lengths else None)
        angles = {}
        # Elbows and knees: angle at the middle keypoint. Values outside broad
        # anatomical image-plane ranges are evidence, not a medical diagnosis.
        for name, a, b, c in (("left_elbow", 5, 7, 9), ("right_elbow", 6, 8, 10),
                              ("left_knee", 11, 13, 15), ("right_knee", 12, 14, 16)):
            if max(a, b, c) < len(points) and min(scores[a], scores[b], scores[c]) > .3:
                u, v = points[a] - points[b], points[c] - points[b]
                denom = max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-8)
                angles[name] = float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1))))
        angle_rows.append(angles)
    speeds = []
    for a, b in zip(centers, centers[1:]):
        if a is not None and b is not None:
            speeds.append(float(np.linalg.norm(b - a) / max(width, height)))
    ratios = np.array([x for x in body_ratios if x is not None])
    angle_changes = []
    angle_anomalies = []
    for index, (before, after) in enumerate(zip(angle_rows, angle_rows[1:])):
        for joint in before.keys() & after.keys():
            delta = abs(after[joint] - before[joint])
            angle_changes.append(delta)
            if delta > 65:
                angle_anomalies.append({"timestamp": rows[index + 1]["timestamp"],
                                        "joint": joint, "delta_degrees": float(delta)})
    return {
        "people_per_frame": people,
        "person_presence_rate": float(np.mean(np.array(people) > 0)),
        "keypoint_visibility_mean": float(np.mean(visibility)),
        "normalized_body_ratio_cv": float(np.std(ratios) / max(np.mean(ratios), 1e-6)) if len(ratios) > 1 else None,
        "normalized_center_speed_mean": float(np.mean(speeds)) if speeds else None,
        "normalized_center_acceleration_mean": float(np.mean(np.abs(np.diff(speeds)))) if len(speeds) > 1 else None,
        "joint_angles_degrees": angle_rows,
        "joint_angle_change_mean": float(np.mean(angle_changes)) if angle_changes else None,
        "joint_angle_jump_rate": float(np.mean(np.asarray(angle_changes) > 65)) if angle_changes else None,
        "joint_angle_anomalies": angle_anomalies,
        "timestamps": [row["timestamp"] for row in rows],
        "source": result["source"], "providers": result["providers"],
    }
