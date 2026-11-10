from pathlib import Path
from unittest.mock import patch

import numpy as np

from video_eval.features import VideoFeatures, objective_signals
from video_eval.pipeline import EvaluationRequest, _run_metric
from video_eval.registry import METRICS


def video_fixture():
    frames = []
    for index in range(4):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[:, :, 1] = 40 + index
        frame[20:80, 20 + index:70 + index] = (40, 120, 220)
        frames.append(frame)
    return VideoFeatures(Path("/tmp/mock.mp4"), 4.0, 4, 100, 100, list(range(4)),
                         frames, [f[:, :, 1] for f in frames], [0., .25, .5, .75])


PARSED = {
    "atoms": [
        {"id": "s", "type": "subject", "text": "a cat", "subjects": ["cat"]},
        {"id": "n", "type": "count", "text": "one cat", "subjects": ["cat"], "target_count": 1},
        {"id": "a", "type": "attribute", "text": "red cat", "subjects": ["cat"], "attributes": ["red"]},
        {"id": "p", "type": "spatial", "text": "cat left of dog", "subjects": ["cat", "dog"], "relation": "left"},
        {"id": "m1", "type": "action", "text": "pick up cup", "subjects": ["person"], "objects": ["cup"], "order": 1},
        {"id": "m2", "type": "action", "text": "place cup", "subjects": ["person"], "objects": ["cup"], "order": 2},
        {"id": "sc", "type": "scene", "text": "outdoors"},
        {"id": "st", "type": "style", "text": "watercolor"},
        {"id": "c", "type": "camera", "text": "fixed camera"},
        {"id": "x", "type": "negative", "text": "no text", "objects": ["text"]},
    ],
    "subject_labels": ["person", "cat", "dog"],
    "object_labels": ["cup", "text"],
    "has_people": True,
    "has_interaction": True,
}


def ground_result(_frames, timestamps, labels):
    rows = []
    for index, timestamp in enumerate(timestamps):
        annotations = []
        for label_index, label in enumerate(labels):
            x1 = 10 + label_index * 18 + index
            if label == "cup":
                x1 = 48 + index * 5
            box = [x1, 35, min(x1 + 12, 99), 65]
            annotations.append({"label": label, "box_xyxy": box, "box_confidence": .9,
                                "mask_score": .9, "mask_area_ratio": .04,
                                "mask_centroid": [(box[0] + box[2]) / 2, 50.]})
        rows.append({"timestamp": timestamp, "annotations": annotations})
    per_label = {label: {"counts": [1] * len(rows), "presence_rate": 1.,
                         "median_count": 1., "count_std": 0.} for label in labels}
    return {"per_label": per_label, "frames": rows, "source": "mock",
            "grounding_checkpoint": "mock", "sam_checkpoint": "mock"}


def pose_raw(timestamps):
    rows = []
    for index, timestamp in enumerate(timestamps):
        points = np.zeros((17, 2), float)
        scores = np.ones(17, float)
        points[:] = [50 + index * 5, 50]
        points[5:17] += np.arange(12)[:, None] * [0.2, 0.3]
        points[9] = [50 + index * 5, 50]
        rows.append({"timestamp": timestamp, "keypoints": np.asarray([points]),
                     "scores": np.asarray([scores])})
    return {"frames": rows, "providers": ["mock"], "source": "mock"}


POSE_SIGNALS = {
    "people_per_frame": [2, 2, 2, 2], "person_presence_rate": 1.,
    "keypoint_visibility_mean": 1., "normalized_body_ratio_cv": .01,
    "normalized_center_speed_mean": .02, "normalized_center_acceleration_mean": .001,
    "joint_angles_degrees": [{}, {}, {}, {}], "joint_angle_change_mean": 2.,
    "joint_angle_jump_rate": 0., "joint_angle_anomalies": [],
    "timestamps": [0., .25, .5, .75], "source": "mock", "providers": ["mock"],
}
TRACKS = {"visibility_rate": .98, "normalized_speed_mean": .03,
          "normalized_speed_std": .005, "normalized_acceleration_mean": .002,
          "jump_rate": .01, "track_count": 10, "frame_count": 4, "source": "mock"}
FACES = {"face_visibility_rate": .75, "identity_similarity_mean": .9,
         "normalized_landmark_error_mean": .01,
         "occlusion_recoveries": [{"similarity": .9, "before": .25, "after": .75}]}


def qwen_answer(_video, metric_id, _name, _prompt, _evidence):
    return {"dimension_id": metric_id, "applicable": True, "score": 4,
            "confidence": .8, "evidence": [{"start": 0., "end": .5, "observation": "mock visible evidence"}],
            "reason": "mock", "needs_review": False}


def test_every_metric_executes_a_real_route_without_fallback():
    video = video_fixture()
    raw = objective_signals(video)
    request = EvaluationRequest(str(video.path), prompt="one red cat left of a dog; a person picks up and places a cup")
    shared = {"prompt": PARSED}
    depth = {"depths": np.ones((4, 100, 100), np.float32), "source": "mock"}
    with patch("video_eval.pipeline.grounded_sam2_adapter.infer", side_effect=ground_result), \
         patch("video_eval.pipeline.grounded_sam2_adapter.summary", side_effect=lambda value: value), \
         patch("video_eval.pipeline.pose_adapter.infer", return_value=pose_raw(video.timestamps)), \
         patch("video_eval.pipeline.pose_adapter.signals", return_value=POSE_SIGNALS), \
         patch("video_eval.pipeline.cotracker_adapter.infer", return_value=TRACKS), \
         patch("video_eval.pipeline.depth_adapter.infer", return_value=depth), \
         patch("video_eval.pipeline.insightface_adapter.infer", return_value=FACES), \
         patch("video_eval.pipeline.vbench_adapter.imaging_quality", return_value={"value": .8}), \
         patch("video_eval.pipeline.vbench_adapter.aesthetic_quality", return_value={"value": .8}), \
         patch("video_eval.pipeline.vbench_adapter.temporal_flickering", return_value={"value": .8}), \
         patch("video_eval.pipeline.vbench_adapter.background_consistency", return_value={"value": .8}), \
         patch("video_eval.pipeline.vbench_adapter.motion_smoothness", return_value={"value": .8}), \
         patch("video_eval.pipeline.vbench_adapter.appearance_style", return_value={"value": .8}), \
         patch("video_eval.pipeline.vbench2_adapter.human_anatomy", return_value={"applicable": True, "value": .8}), \
         patch("video_eval.pipeline.vbench2_adapter.human_identity", return_value={"applicable": True, "value": .8}), \
         patch("video_eval.pipeline.vbench2_adapter.human_clothes", return_value={"applicable": True, "value": .8}), \
         patch("video_eval.pipeline.vbench2_adapter.multi_view_consistency", return_value={"applicable": True, "value": .8}), \
         patch("video_eval.pipeline.judge", side_effect=qwen_answer):
        results = [_run_metric(metric, video, raw, request, shared) for metric in METRICS]
    assert len(results) == 18
    assert not [r for r in results if "backend adapter not configured" in r.reason]
    assert not [r for r in results if r.status == "ERROR"]
    assert all(r.status in {"SUCCESS", "UNCALIBRATED", "NOT_APPLICABLE",
                            "INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW"} for r in results)
    # Every implemented route may emit a score, but it must stay on the public
    # 1-5 scale and retain an explicit maturity/status marker.
    for metric, result in zip(METRICS, results):
        if result.score is not None:
            assert 1 <= result.score <= 5
            assert result.implementation_status


if __name__ == "__main__":
    test_every_metric_executes_a_real_route_without_fallback()
    print("all-metric-execution=18/18")
