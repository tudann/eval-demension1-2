from __future__ import annotations

import copy

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .features import decode, objective_signals
from .qwen import QwenUnavailable, judge
from .registry import MetricSpec, select
from .schema import Evidence, MetricResult
from . import vbench_adapter
from . import pose_adapter
from . import cotracker_adapter
from .prompt_parser import parse as parse_prompt, relevant as relevant_atoms
from . import grounded_sam2_adapter
from . import depth_adapter
from . import insightface_adapter
from . import vbench2_adapter
from . import interaction


@dataclass(slots=True)
class EvaluationRequest:
    video_path: str
    prompt: str | None = None
    reference_images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled_metrics: list[str] | None = None
    enabled_dimensions: list[int] | None = None
    sample_fps: float = 4.0
    max_frames: int = 96


def map_quality_to_score(quality: float) -> float:
    """User-authorized linear map from a higher-is-better 0–1 quality to 1–5.

    This is not a human-label calibration. 0 maps to 1 and 1 maps to 5.
    """
    clipped = float(np.clip(quality, 0.0, 1.0))
    return round(1.0 + 4.0 * clipped, 2)


def _with_mapped_score(raw_metrics: dict[str, Any]) -> tuple[dict[str, Any], float | None]:
    raw_metrics = dict(raw_metrics)
    quality = raw_metrics.get("uncalibrated_quality_0_1")
    if quality is None:
        for key in ("subject_presence_quality_0_1", "t2v_exact_frame_match_0_1"):
            value = raw_metrics.get(key)
            if isinstance(value, (int, float)):
                quality = float(value)
                break
    if quality is None:
        value = raw_metrics.get("value")
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            quality = float(value)
    if quality is None:
        return raw_metrics, None
    score = map_quality_to_score(float(quality))
    raw_metrics["calibration_version"] = "linear_0_1_to_1_5_v1"
    raw_metrics["mapped_score_1_5"] = score
    return raw_metrics, score


def _objective(metric: MetricSpec, raw: dict[str, Any]) -> MetricResult | None:
    # Thresholds are deliberately labelled UNCALIBRATED until fitted on human labels.
    formulas = {
        "D2.1": (np.clip(np.log1p(raw["sharpness_p10"]) / np.log(501), 0, 1), ["sharpness_mean", "sharpness_p10"]),
        "D2.2": (np.clip((raw["entropy_mean"] - 3.0) / 4.5, 0, 1), ["entropy_mean", "sharpness_mean"]),
        "D2.5": (np.exp(-abs(raw["brightness_mean"] - 127.5) / 90) * np.clip(raw["contrast_mean"] / 55, 0, 1), ["brightness_mean", "contrast_mean"]),
        "D3.1": (np.exp(-raw["frame_residual_std"] / 25), ["frame_residual_mean", "frame_residual_std"]),
        "D3.2": (np.exp(-raw["brightness_std"] / 18), ["brightness_std"]),
        "D3.4": (np.exp(-raw["frame_residual_std"] / 20), ["frame_residual_std"]),
        "D3.5": (np.exp(-raw["brightness_std"] / 15), ["brightness_mean", "brightness_std"]),
        "D5.2": (np.exp(-raw["flow_acceleration"] / 3), ["flow_mean", "flow_std", "flow_acceleration"]),
        "D5.5": (np.exp(-raw["flow_acceleration"] / 5), ["flow_mean", "flow_acceleration"]),
        "D5.6": (np.exp(-raw["flow_std"] / 8), ["flow_mean", "flow_std"]),
    }
    if metric.id not in formulas:
        return None
    quality, keys = formulas[metric.id]
    values = {key: raw[key] for key in keys}
    values["uncalibrated_quality_0_1"] = float(quality)
    values, score = _with_mapped_score(values)
    return MetricResult(metric.id, metric.name, metric.kind, "UNCALIBRATED", "UNCALIBRATED", True,
                        score, 0.55, values, [],
                        "Objective proxy mapped to 1-5 with linear_0_1_to_1_5_v1 (0→1, 1→5). Not fitted on human labels.", True)


def _unavailable(metric: MetricSpec, reason: str) -> MetricResult:
    return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status,
                        "MODEL_UNAVAILABLE", None, reason=reason, needs_review=True)


def _insufficient(metric: MetricSpec, reason: str, raw_metrics: dict[str, Any] | None = None) -> MetricResult:
    return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status,
                        "INSUFFICIENT_EVIDENCE", None, raw_metrics=raw_metrics or {},
                        reason=reason, needs_review=True)


def _checklist(metric: MetricSpec, video, request: EvaluationRequest, raw_metrics: dict[str, Any],
               fallback_reason: str) -> MetricResult:
    """Ask one OmniVBench-style question. Keep a detector map only if the judge fails."""
    raw_metrics, mapped = _with_mapped_score(dict(raw_metrics))
    try:
        answer = judge(video, metric.id, metric.name, request.prompt, raw_metrics)
    except Exception as exc:
        if mapped is None:
            result = _unavailable(metric, "checklist judge failed")
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        return _uncalibrated(metric, raw_metrics, .7,
                             fallback_reason + f" Judge failed ({type(exc).__name__}); kept the detector map.")
    evidence = [Evidence(**item) for item in answer.get("evidence", [])]
    raw_metrics["checklist"] = "omnivbench_single_question_v1"
    return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status,
                        "NEEDS_REVIEW" if answer.get("needs_review") else "SUCCESS",
                        answer.get("applicable"), answer.get("score"), answer.get("confidence"),
                        raw_metrics, evidence, answer.get("reason", ""),
                        bool(answer.get("needs_review")))


def _uncalibrated(metric: MetricSpec, raw_metrics: dict[str, Any], confidence: float,
                  reason: str) -> MetricResult:
    raw_metrics, score = _with_mapped_score(raw_metrics)
    if score is None:
        raw_metrics.setdefault("calibration_version", None)
        note = reason
    else:
        note = reason + " Mapped to 1-5 with linear_0_1_to_1_5_v1 (0→1, 1→5)."
    return MetricResult(metric.id, metric.name, metric.kind, "UNCALIBRATED",
                        "UNCALIBRATED", True, score, confidence, raw_metrics, [],
                        note, True)


def _run_metric(metric: MetricSpec, video, raw, request: EvaluationRequest, shared: dict[str, Any]) -> MetricResult:
    if metric.prompt_required and not request.prompt:
        return _insufficient(metric, "prompt is required for this metric")
    atoms = None
    if request.prompt and (metric.id.startswith("D1.") or metric.id == "D5.7" or metric.id.startswith("D6.")) and "prompt" not in shared and "prompt_error" not in shared:
        try:
            shared["prompt"] = parse_prompt(request.prompt)
        except Exception as exc:
            shared["prompt_error"] = f"{type(exc).__name__}: {exc}"
    if metric.id.startswith("D1.") and "prompt_error" in shared and "prompt" not in shared:
        return _insufficient(metric, f"prompt parse failed once for this video: {shared['prompt_error']}")
    if metric.id.startswith("D1."):
        atoms = relevant_atoms(shared["prompt"], metric.id)
        # These clips were selected for the dimension, so a missing parser atom
        # is not evidence that the case fails the dimension. Count, style, and
        # negation are optional sub-requirements and can still be absent.
        if metric.id in {"D1.2", "D1.8", "D1.10"} and not atoms:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status,
                                "NOT_APPLICABLE", False, raw_metrics={"prompt_atoms": []},
                                reason="Prompt does not contain this optional constraint.")
        if not atoms:
            atoms = [{"text": request.prompt, "source": "full_prompt"}]
    grounding_metrics = {"D1.1", "D1.2", "D1.3", "D1.4", "D1.6", "D1.10", "D1.11",
                         "D3.3", "D3.4", "D3.5", "D4.3", "D4.5", "D4.6", "D4.8",
                         "D5.7", "D6.1", "D6.2", "D6.3", "D6.4", "D6.5", "D6.6", "D6.7", "D6.8"}
    grounding_evidence = None
    if metric.id in grounding_metrics:
        labels = []
        if "prompt" in shared:
            labels = shared["prompt"].get("subject_labels", [])
            if metric.id.startswith("D6.") or metric.id in {"D1.10", "D1.11"}:
                labels = list(labels) + list(shared["prompt"].get("object_labels", []))
        if not labels and metric.id.startswith(("D4.", "D5.", "D6.")):
            labels = ["person"]
        if metric.id.startswith("D6."):
            # Parser object lists are often empty. Querying only the person
            # makes every later object check fail.
            labels = list(dict.fromkeys(list(labels) + ["person", "hand", "object"]))
        if metric.id == "D4.3":
            labels = list(dict.fromkeys(list(labels) + ["person", "hair", "hairstyle"]))
        elif metric.id == "D4.4":
            labels = list(dict.fromkeys(list(labels) + ["person", "clothing", "shirt", "pants", "dress"]))
        elif metric.id == "D4.5":
            labels = list(dict.fromkeys(list(labels) + ["person", "glasses", "hat", "earring", "necklace", "accessory"]))
        if labels:
            key = "grounding:" + "|".join(labels)
            if key not in shared:
                shared[key] = grounded_sam2_adapter.summary(
                    grounded_sam2_adapter.infer(video.frames, video.timestamps, labels))
            grounding_evidence = shared[key]
    # Compute shared prerequisites before any scorer consumes them.  The old
    # implementation used tracker_evidence in D3.3/D3.4 before assigning it.
    pose_metrics = {"D1.5", "D4.6", "D5.1", "D5.3", "D5.4", "D5.8",
                    "D6.1", "D6.2", "D6.5", "D6.6", "D6.7"}
    tracker_metrics = {"D1.9", "D3.1", "D3.3", "D3.4", "D5.5", "D5.6", "D5.8", "D6.2", "D6.3", "D6.8"}
    pose_evidence = None
    pose_raw = None
    if metric.id in pose_metrics:
        if "pose_raw" not in shared:
            shared["pose_raw"] = pose_adapter.infer(video.frames, video.timestamps)
            shared["pose"] = pose_adapter.signals(shared["pose_raw"], video.width, video.height)
        pose_raw, pose_evidence = shared["pose_raw"], shared["pose"]
    tracker_evidence = None
    if metric.id in tracker_metrics:
        if "tracker" not in shared:
            shared["tracker"] = cotracker_adapter.infer(video.frames, video.width, video.height)
        tracker_evidence = shared["tracker"]
    if metric.id == "D1.1":
        if grounding_evidence is None:
            return _insufficient(metric, "prompt parser produced no detectable subject labels")
        rates = [item["presence_rate"] for item in grounding_evidence["per_label"].values()]
        quality = float(np.mean(rates)) if rates else 0.0
        missing = [label for label, item in grounding_evidence["per_label"].items() if item["presence_rate"] < .5]
        evidence = {**grounding_evidence, "uncalibrated_quality_0_1": quality,
                    "subject_presence_quality_0_1": quality,
                    "low_presence_subjects": missing}
        return _checklist(metric, video, request, evidence,
                          "Grounded-SAM-2 subject presence measured.")
    if metric.id == "D1.2":
        count_atoms = [atom for atom in atoms or [] if atom.get("target_count") is not None]
        if not count_atoms or grounding_evidence is None:
            return _insufficient(metric, "no count constraint or detectable subject label")
        comparisons = []
        per_frame_matches = []
        for atom in count_atoms:
            target = int(atom["target_count"])
            subject = next(iter(atom.get("subjects") or []), None)
            matched = next((value for label, value in grounding_evidence["per_label"].items()
                            if subject and subject.lower() in label.lower()), None)
            if matched:
                error = abs(matched["median_count"] - target)
                comparisons.append({"subject": subject, "target": target, **matched, "absolute_error": error})
                per_frame_matches.append([int(count == target) for count in matched["counts"]])
        if not comparisons:
            return _insufficient(metric, "count atom could not be aligned to Grounded-SAM-2 labels")
        # T2V-CompBench Generative Numeracy: each sampled frame receives the
        # fraction of requested object counts that match exactly; video score
        # is the mean frame score. Count variance is reported, not mixed into
        # that official score.
        quality = float(np.mean(np.asarray(per_frame_matches, dtype=float), axis=0).mean())
        evidence = {"comparisons": comparisons, "detector": grounding_evidence,
                    "uncalibrated_quality_0_1": quality,
                    "t2v_exact_frame_match_0_1": quality}
        return _checklist(metric, video, request, evidence,
                          "T2V-CompBench exact per-frame numeracy aggregation.")
    if metric.id == "D1.4":
        if grounding_evidence is None:
            return _insufficient(metric, "spatial evaluation requires detected subjects")
        if "depth" not in shared:
            shared["depth"] = depth_adapter.infer(video.frames, max(1.0, request.sample_fps))
        depths = shared["depth"]["depths"]
        checks = []
        for atom in atoms or []:
            subjects = atom.get("subjects") or []
            relation = str(atom.get("relation") or atom.get("text") or "").lower()
            if len(subjects) < 2:
                continue
            for frame_index, row in enumerate(grounding_evidence["frames"]):
                found = []
                for subject in subjects[:2]:
                    found.append(next((item for item in row["annotations"] if subject.lower() in item["label"].lower()), None))
                if None in found:
                    continue
                first, second = found
                x1, y1 = first["mask_centroid"]
                x2, y2 = second["mask_centroid"]
                d1 = float(depths[frame_index, min(int(y1), depths.shape[1]-1), min(int(x1), depths.shape[2]-1)])
                d2 = float(depths[frame_index, min(int(y2), depths.shape[1]-1), min(int(x2), depths.shape[2]-1)])
                if any(word in relation for word in ("left", "左")): passed = x1 < x2
                elif any(word in relation for word in ("right", "右")): passed = x1 > x2
                elif any(word in relation for word in ("above", "over", "上")): passed = y1 < y2
                elif any(word in relation for word in ("below", "under", "下")): passed = y1 > y2
                elif any(word in relation for word in ("front", "前")): passed = d1 > d2
                elif any(word in relation for word in ("behind", "后")): passed = d1 < d2
                else: continue
                checks.append({"timestamp": row["timestamp"], "subjects": subjects[:2], "relation": relation,
                               "centroids": [[x1, y1], [x2, y2]], "relative_depth": [d1, d2], "passed": bool(passed)})
        if not checks:
            return _insufficient(metric, "no spatial relation could be evaluated on jointly visible subjects")
        quality = float(np.mean([item["passed"] for item in checks]))
        raw_depth = {key: value for key, value in shared["depth"].items() if key != "depths"}
        return _uncalibrated(metric, {"checks": checks, "depth": raw_depth,
                                      "relation_accuracy_0_1": quality}, .75,
                             "Grounded-SAM-2 centroids plus relative depth; 1-5 calibration is pending.")
    if metric.id in {"D3.3", "D3.4", "D3.5"}:
        if grounding_evidence is None:
            return _insufficient(metric, "no trackable subject labels were available")
        selected = []
        for frame, row in zip(video.frames, grounding_evidence["frames"]):
            if not row["annotations"]:
                selected.append(None)
                continue
            item = max(row["annotations"], key=lambda value: value["box_confidence"])
            x1, y1, x2, y2 = [int(round(value)) for value in item["box_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(video.width, x2), min(video.height, y2)
            crop = frame[y1:y2, x1:x2]
            selected.append((item, crop) if crop.size else None)
        valid = [item for item in selected if item is not None]
        if len(valid) < 2:
            return _insufficient(metric, "subject is visible in fewer than two sampled frames")
        if metric.id == "D3.3":
            areas = np.array([item[0]["mask_area_ratio"] for item in valid])
            aspects = np.array([(item[0]["box_xyxy"][2] - item[0]["box_xyxy"][0]) /
                                max(item[0]["box_xyxy"][3] - item[0]["box_xyxy"][1], 1) for item in valid])
            area_cv = float(np.std(areas) / max(np.mean(areas), 1e-6))
            aspect_cv = float(np.std(aspects) / max(np.mean(aspects), 1e-6))
            quality = np.exp(-3 * area_cv - 4 * aspect_cv) * tracker_evidence["visibility_rate"]
            metrics = {"mask_area_cv": area_cv, "box_aspect_cv": aspect_cv, "tracks": tracker_evidence,
                       "timestamps": [row["timestamp"] for row in grounding_evidence["frames"]]}
            metrics["uncalibrated_quality_0_1"] = float(quality)
            return _uncalibrated(metric, metrics, .75,
                                 "Grounded-SAM-2 geometry fused with CoTracker3; calibration is pending.")
        if metric.id == "D3.4":
            histograms = []
            for _, crop in valid:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                histograms.append(cv2.normalize(hist, hist).flatten())
            correlations = [float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL)) for a, b in zip(histograms, histograms[1:])]
            quality = float(np.clip(np.mean(correlations), 0, 1)) * tracker_evidence["visibility_rate"]
            metrics = {"adjacent_texture_histogram_correlation": correlations, "tracks": tracker_evidence}
            metrics["uncalibrated_quality_0_1"] = float(quality)
            return _uncalibrated(metric, metrics, .7,
                                 "Tracked-region texture correlation measured; calibration is pending.")
        brightness = [float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) for _, crop in valid]
        delta = float(np.std(brightness))
        quality = float(np.exp(-delta / 15))
        return _uncalibrated(metric, {"region_brightness": brightness, "brightness_std": delta,
                                      "uncalibrated_quality_0_1": float(quality)}, .75,
                             "Region brightness stability measured; calibration is pending.")
    if metric.id in {"D4.2", "D4.8"}:
        if "faces" not in shared:
            shared["faces"] = insightface_adapter.infer(video.frames, video.timestamps)
        faces = shared["faces"]
        if faces["face_visibility_rate"] == 0:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics=faces, reason="InsightFace detected no usable face.")
        if metric.id == "D4.2":
            similarity, landmark_error = faces["identity_similarity_mean"], faces["normalized_landmark_error_mean"]
            if similarity is None or landmark_error is None:
                return _insufficient(metric, "insufficient consecutive visible faces", faces)
            quality = np.clip((similarity-.5)/.5, 0, 1) * np.exp(-12*landmark_error)
            evidence = {**faces, "uncalibrated_quality_0_1": float(quality)}
            return _uncalibrated(metric, evidence, .82,
                                 "InsightFace embedding and 106-point shape measured; calibration is pending.")
        recoveries = faces["occlusion_recoveries"]
        if not recoveries:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics=faces, reason="No face occlusion-and-reappearance event was observed.")
        quality = float(np.mean([np.clip((item["similarity"]-.5)/.5, 0, 1) for item in recoveries]))
        combined = {"faces": faces, "grounded_subjects": grounding_evidence}
        combined["uncalibrated_quality_0_1"] = quality
        return _uncalibrated(metric, combined, .8,
                             "Pre/post-occlusion identity similarity measured; calibration is pending.")
    if metric.id == "D2.3":
        if "human_anatomy" not in shared:
            shared["human_anatomy"] = vbench2_adapter.human_anatomy(str(video.path))
        measured = copy.deepcopy(shared["human_anatomy"])
        if not measured["applicable"]:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics=measured, reason=measured["reason"])
        return _uncalibrated(metric, measured, .9,
                             "Official VBench-2.0 Human Anatomy executed; mapping needs calibration.")
    if metric.id == "D4.1":
        measured = vbench2_adapter.human_identity(str(video.path))
        if not measured["applicable"]:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics=measured, reason=measured["reason"])
        return _uncalibrated(metric, measured, .9,
                             "Official VBench-2.0 Human Identity executed; mapping needs calibration.")
    if metric.id == "D4.4":
        measured = vbench2_adapter.human_clothes(str(video.path))
        if not measured["applicable"]:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics=measured, reason=measured["reason"])
        return _uncalibrated(metric, measured, .85,
                             "Official VBench-2.0 Human Clothes executed; mapping needs calibration.")
    if metric.id == "D4.7":
        measured = vbench2_adapter.multi_view_consistency(str(video.path))
        if not measured["applicable"]:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics=measured, reason=measured["reason"])
        # Official raw range is not assumed to be [0,1].
        return MetricResult(metric.id, metric.name, metric.kind, "UNCALIBRATED", "UNCALIBRATED", True,
                            None, .85, measured, [], "Official VBench-2.0 Multi-View output; no 1-5 calibration yet.", True)
    if metric.id == "D1.8":
        style_text = "; ".join(str(atom.get("text", "")) for atom in atoms or [] if atom.get("text"))
        if not style_text:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                reason="Prompt contains no appearance-style constraint.")
        shared["appearance_style"] = vbench_adapter.appearance_style(str(video.path), style_text)
    if metric.id == "D5.4":
        visible = pose_evidence["keypoint_visibility_mean"]
        accel = pose_evidence["normalized_center_acceleration_mean"]
        if pose_evidence["person_presence_rate"] == 0:
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics=pose_evidence, reason="No person detected by DWPose.")
        jump_rate = pose_evidence["joint_angle_jump_rate"]
        if jump_rate is None:
            return _insufficient(metric, "Insufficient consecutive elbow/knee keypoints for joint-angle analysis.",
                                 pose_evidence)
        quality = visible * np.exp(-10 * (accel or 0.0)) * (1.0 - jump_rate)
        evidence = {**pose_evidence, "uncalibrated_quality_0_1": float(quality)}
        return _uncalibrated(metric, evidence, .8,
                             "DWPose joint-angle and trajectory evidence measured; calibration is pending.")
    if metric.id == "D4.6":
        cv = pose_evidence["normalized_body_ratio_cv"]
        if cv is None:
            return _insufficient(metric, "insufficient visible body keypoints across frames", pose_evidence)
        mask_aspects = []
        if grounding_evidence:
            for row in grounding_evidence["frames"]:
                people = [item for item in row["annotations"] if "person" in item["label"].lower()]
                if people:
                    box = max(people, key=lambda item: item["box_confidence"])["box_xyxy"]
                    mask_aspects.append((box[2] - box[0]) / max(box[3] - box[1], 1.0))
        mask_aspect_cv = (float(np.std(mask_aspects) / max(np.mean(mask_aspects), 1e-6))
                          if len(mask_aspects) > 1 else None)
        quality = float(np.exp(-8 * cv - (4 * mask_aspect_cv if mask_aspect_cv is not None else 0)))
        evidence = {**pose_evidence, "grounded_person_box_aspects": mask_aspects,
                    "grounded_person_box_aspect_cv": mask_aspect_cv,
                    "uncalibrated_quality_0_1": quality}
        return _uncalibrated(metric, evidence, .75,
                             "DWPose body geometry fused with Grounded-SAM-2 person proportions; calibration is pending.")
    if metric.id == "D5.8" and pose_evidence["person_presence_rate"] > 0:
        motion = tracker_evidence["normalized_speed_mean"] or 0.0
        quality = pose_evidence["keypoint_visibility_mean"] * tracker_evidence["visibility_rate"] * min(1.0, motion / .01)
        combined = {"pose": pose_evidence, "tracks": tracker_evidence}
        combined["uncalibrated_quality_0_1"] = float(quality)
        return _uncalibrated(metric, combined, .75,
                             "High-dynamic pose/track evidence measured; calibration is pending.")
    if metric.id == "D5.8":
        return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status,
                            "NOT_APPLICABLE", False, raw_metrics={"pose": pose_evidence, "tracks": tracker_evidence},
                            reason="No person detected for high-dynamic human-motion evaluation.")
    if metric.id == "D3.1":
        smooth = vbench_adapter.motion_smoothness(str(video.path))["value"]
        jump = tracker_evidence["jump_rate"] or 0.0
        quality = smooth * tracker_evidence["visibility_rate"] * (1.0 - jump)
        combined = {"vbench_motion_smoothness": smooth, "tracks": tracker_evidence}
        combined["uncalibrated_quality_0_1"] = float(quality)
        return _uncalibrated(metric, combined, .85,
                             "VBench smoothness fused with CoTracker3; calibration is pending.")
    if metric.id == "D5.5":
        speed = tracker_evidence["normalized_speed_mean"]
        accel = tracker_evidence["normalized_acceleration_mean"]
        if speed is None or accel is None:
            return _insufficient(metric, "insufficient visible CoTracker3 trajectories", tracker_evidence)
        irregularity = accel / max(speed, 1e-5)
        evidence = {**tracker_evidence, "relative_speed_irregularity": float(irregularity),
                    "uncalibrated_quality_0_1": float(np.exp(-irregularity))}
        return _uncalibrated(metric, evidence, .8,
                             "Relative pixel-speed regularity measured; no physical-speed claim.")
    if metric.id == "D5.6":
        jump = tracker_evidence["jump_rate"]
        if jump is None:
            return _insufficient(metric, "insufficient visible CoTracker3 trajectories", tracker_evidence)
        quality = tracker_evidence["visibility_rate"] * (1.0 - jump)
        evidence = {**tracker_evidence, "uncalibrated_quality_0_1": float(quality)}
        return _uncalibrated(metric, evidence, .82,
                             "CoTracker3 visibility and jump rate measured; calibration is pending.")
    if metric.id.startswith("D6."):
        prompt_info = shared.get("prompt")
        if not interaction.interaction_applicable(prompt_info, grounding_evidence):
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status, "NOT_APPLICABLE", False,
                                raw_metrics={"prompt_has_interaction": bool(prompt_info and prompt_info.get("has_interaction"))},
                                reason="No requested or detected person-object/multi-person interaction.")
        interaction_evidence: dict[str, Any] = {}
        if pose_raw is not None and grounding_evidence is not None:
            interaction_evidence["contact"] = interaction.contact_signals(
                pose_raw, grounding_evidence, video.width, video.height)
        if grounding_evidence is not None:
            interaction_evidence["object_motion"] = interaction.object_motion(
                grounding_evidence, video.width, video.height)
        shared["interaction"] = interaction_evidence
        if metric.id == "D6.1":
            contact = interaction_evidence.get("contact", {})
            rate = contact.get("candidate_contact_rate")
            if rate is None:
                return _insufficient(metric, "No jointly visible body keypoints and interaction object.",
                                     interaction_evidence)
            evidence = {**interaction_evidence, "uncalibrated_quality_0_1": float(rate)}
            return _uncalibrated(metric, evidence, .72,
                                 "Candidate contact rate from 2-D keypoint/box proximity. Not proof of physical contact.")
        if metric.id == "D6.3":
            objects = interaction_evidence.get("object_motion", {}).get("objects", {})
            valid = {k: v for k, v in objects.items() if v["observations"] >= 3}
            if not valid:
                return _insufficient(metric, "No interaction object has enough observations for pickup/place state analysis.",
                                     interaction_evidence)
            staged = {k: {**v, "moved": v["total_displacement_norm"] > .03,
                           "placed_stably": v["tail_stability"] is not None and v["tail_stability"] < .01}
                      for k, v in valid.items()}
            tails = [v["tail_stability"] for v in staged.values() if v["tail_stability"] is not None]
            moved = [1.0 if v["moved"] else 0.0 for v in staged.values()]
            stable = [float(np.exp(-tail / 0.05)) for tail in tails]
            quality = 0.5 * float(np.mean(moved)) + 0.5 * float(np.mean(stable) if stable else 0.0)
            return _checklist(metric, video, request,
                              {"objects": staged, "contact": interaction_evidence.get("contact"),
                               "tracks": tracker_evidence, "uncalibrated_quality_0_1": quality},
                              "Pickup/place detector evidence retained.")
        if metric.id == "D6.8":
            objects = interaction_evidence.get("object_motion", {}).get("objects", {})
            tails = [v["tail_stability"] for v in objects.values() if v["tail_stability"] is not None]
            if not tails:
                return _insufficient(metric, "No post-interaction object track with a measurable tail.",
                                     interaction_evidence)
            quality = float(np.mean([np.exp(-tail / 0.05) for tail in tails]))
            return _uncalibrated(metric, {**interaction_evidence, "tracks": tracker_evidence,
                                           "uncalibrated_quality_0_1": quality}, .72,
                                 "Post-event tail stability mapped from displacement. Lower drift scores higher.")
    official_vbench = {
        "D2.1": vbench_adapter.imaging_quality,
        "D2.7": vbench_adapter.aesthetic_quality,
        "D3.2": vbench_adapter.temporal_flickering,
        "D3.6": vbench_adapter.background_consistency,
        "D5.2": vbench_adapter.motion_smoothness,
    }
    if metric.id in official_vbench:
        measured = official_vbench[metric.id](str(video.path))
        raw_value = measured["value"]
        measured["official_raw_value"] = raw_value
        return _uncalibrated(metric, measured, .9,
                             "Official VBench metric executed; 1-5 mapping remains pending calibration.")
    objective = _objective(metric, raw)
    if metric.backend == "qwen" or "+qwen" in metric.backend:
        try:
            evidence_input = dict(objective.raw_metrics if objective else {})
            if atoms is not None:
                evidence_input["prompt_atoms"] = atoms
            if pose_evidence:
                evidence_input["pose"] = pose_evidence
            if tracker_evidence:
                evidence_input["tracks"] = tracker_evidence
            if grounding_evidence:
                evidence_input["grounded_masks"] = grounding_evidence
            if metric.id == "D1.8" and "appearance_style" in shared:
                evidence_input["vbench_appearance_style"] = shared["appearance_style"]
            if metric.id == "D2.6":
                if "human_anatomy" not in shared:
                    shared["human_anatomy"] = vbench2_adapter.human_anatomy(str(video.path))
                evidence_input["vbench2_human_anatomy"] = shared["human_anatomy"]
            if metric.id.startswith("D6."):
                evidence_input["interaction"] = shared.get("interaction", {})
            answer = judge(video, metric.id, metric.name, request.prompt, evidence_input)
            evidence = [Evidence(**item) for item in answer.get("evidence", [])]
            return MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status,
                                "NEEDS_REVIEW" if answer.get("needs_review") else "SUCCESS",
                                answer.get("applicable"), answer.get("score"), answer.get("confidence"),
                                evidence_input, evidence, answer.get("reason", ""),
                                bool(answer.get("needs_review")))
        except QwenUnavailable as exc:
            return objective or _unavailable(metric, str(exc))
        except Exception as exc:
            result = objective or _unavailable(metric, "Qwen inference failed")
            result.error = f"{type(exc).__name__}: {exc}"
            return result
    if objective:
        return objective
    return _unavailable(metric, f"backend adapter not configured: {metric.backend}")


def evaluate(request: EvaluationRequest) -> dict[str, Any]:
    started = time.perf_counter()
    video = decode(request.video_path, request.sample_fps, request.max_frames)
    raw = objective_signals(video)
    metrics = select(request.enabled_metrics, request.enabled_dimensions)
    results = []
    shared: dict[str, Any] = {}
    for metric in metrics:
        print(f"METRIC_START {metric.id}", flush=True)
        try:
            results.append(_run_metric(metric, video, raw, request, shared))
        except Exception as exc:
            results.append(MetricResult(metric.id, metric.name, metric.kind, metric.implementation_status,
                                        "ERROR", None, reason="metric execution failed", needs_review=True,
                                        error=f"{type(exc).__name__}: {exc}"))
        print(f"METRIC_END {metric.id} {results[-1].status} {results[-1].error or chr(32)}", flush=True)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "request": asdict(request),
        "video": {"path": str(video.path), "fps": video.fps, "frame_count": video.frame_count,
                  "duration": video.duration, "width": video.width, "height": video.height,
                  "sample_indices": video.sample_indices, "sample_timestamps": video.timestamps},
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__,
                    "elapsed_seconds": round(time.perf_counter() - started, 3)},
        "summary": {"requested": len(results), "status_counts": counts,
                    "scored": sum(result.score is not None for result in results)},
        "metrics": [result.to_dict() for result in results],
    }
