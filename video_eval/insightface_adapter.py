from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import os

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _app(use_gpu: bool):
    use_gpu = use_gpu and os.getenv("VIDEO_EVAL_ONNX_DEVICE", "cuda") != "cpu"
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    app = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models" / "insightface"), providers=providers, sess_options=options)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    return app


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-8))


def infer(frames: list[np.ndarray], timestamps: list[float], use_gpu: bool = True) -> dict:
    rows = []
    for frame, timestamp in zip(frames, timestamps):
        faces = _app(use_gpu).get(frame)
        if not faces:
            rows.append({"timestamp": timestamp, "visible": False})
            continue
        face = max(faces, key=lambda item: float((item.bbox[2]-item.bbox[0]) * (item.bbox[3]-item.bbox[1])))
        landmarks = getattr(face, "landmark_2d_106", None)
        if landmarks is None:
            landmarks = face.kps
        landmarks = np.asarray(landmarks, dtype=np.float32)
        center = landmarks.mean(axis=0)
        scale = max(float(np.linalg.norm(landmarks.max(axis=0)-landmarks.min(axis=0))), 1e-6)
        shape = ((landmarks-center)/scale).reshape(-1)
        rows.append({"timestamp": timestamp, "visible": True, "bbox": face.bbox.tolist(),
                     "det_score": float(face.det_score), "embedding": face.normed_embedding.astype(np.float32),
                     "landmark_shape": shape})
    visible = [row for row in rows if row["visible"]]
    similarities, shape_errors = [], []
    for first, second in zip(visible, visible[1:]):
        similarities.append(_cosine(first["embedding"], second["embedding"]))
        if first["landmark_shape"].shape == second["landmark_shape"].shape:
            shape_errors.append(float(np.mean(np.abs(first["landmark_shape"]-second["landmark_shape"]))))
    recoveries = []
    for index in range(1, len(rows)-1):
        if rows[index-1]["visible"] and not rows[index]["visible"]:
            later = next((row for row in rows[index+1:] if row["visible"]), None)
            if later:
                recoveries.append({"before": rows[index-1]["timestamp"], "after": later["timestamp"],
                                   "similarity": _cosine(rows[index-1]["embedding"], later["embedding"])})
    serial_rows = [{key: value for key, value in row.items() if key not in {"embedding", "landmark_shape"}} for row in rows]
    return {"face_visibility_rate": len(visible)/len(rows), "adjacent_identity_similarity": similarities,
            "identity_similarity_mean": float(np.mean(similarities)) if similarities else None,
            "normalized_landmark_error_mean": float(np.mean(shape_errors)) if shape_errors else None,
            "occlusion_recoveries": recoveries, "frames": serial_rows,
            "execution_device": os.getenv("VIDEO_EVAL_ONNX_DEVICE", "cuda"), "source": "InsightFace buffalo_l", "model_root": str(ROOT / "models/insightface/buffalo_l")}
