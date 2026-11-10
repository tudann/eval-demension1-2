from pathlib import Path

from video_eval.registry import METRICS


def test_all_metrics_have_non_scaffolded_routes():
    assert len(METRICS) == 18
    assert all(m.implementation_status != "SCAFFOLDED" for m in METRICS)
    source = (Path(__file__).resolve().parents[1] / "video_eval/pipeline.py").read_text()
    routed = set()
    for metric in METRICS:
        if metric.id in source or metric.backend == "qwen" or "+qwen" in metric.backend:
            routed.add(metric.id)
    assert routed == {m.id for m in METRICS}


if __name__ == "__main__":
    test_all_metrics_have_non_scaffolded_routes()
    print("coverage=18/18 routed; status is not claimed IMPLEMENTED")
