from video_eval.registry import METRICS


def test_registry_has_18_unique_metrics():
    assert len(METRICS) == 18
    assert len({metric.id for metric in METRICS}) == 18
    assert [sum(metric.id.startswith(f"D{i}.") for metric in METRICS) for i in range(1, 3)] == [11, 7]


def test_only_approved_qwen_name_is_used():
    from video_eval.qwen import MODEL
    assert MODEL == "Qwen3-VL-32B-Instruct"
