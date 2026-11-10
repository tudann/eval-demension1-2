import numpy as np

from video_eval.interaction import contact_signals, object_motion


def fixture(contact=True):
    wrist_x = 50 if contact else 5
    points = np.zeros((17, 2), dtype=float)
    scores = np.zeros(17, dtype=float)
    points[9] = [wrist_x, 50]
    scores[9] = 1
    pose = {"frames": [{"timestamp": float(i), "keypoints": np.asarray([points]),
                         "scores": np.asarray([scores])} for i in range(3)]}
    grounding = {"frames": [{"timestamp": float(i), "annotations": [
        {"label": "person", "box_xyxy": [0, 0, 100, 100], "mask_centroid": [50, 50]},
        {"label": "cup", "box_xyxy": [45 + i * 5, 45, 55 + i * 5, 55],
         "mask_centroid": [50 + i * 5, 50]},
    ]} for i in range(3)]}
    return pose, grounding


def test_contact_positive_and_negative():
    pose, grounding = fixture(True)
    assert contact_signals(pose, grounding, 100, 100)["candidate_contact_frames"] >= 1
    pose, grounding = fixture(False)
    assert contact_signals(pose, grounding, 100, 100)["candidate_contact_frames"] == 0


def test_object_motion_and_tail():
    _, grounding = fixture(True)
    result = object_motion(grounding, 100, 100)["objects"]["cup"]
    assert result["observations"] == 3
    assert result["total_displacement_norm"] > 0


if __name__ == "__main__":
    test_contact_positive_and_negative()
    test_object_motion_and_tail()
    print("interaction=ok")
