from video_eval.registry import METRICS
from pathlib import Path

assert len(METRICS) == 18
assert len({metric.id for metric in METRICS}) == 18
assert [sum(metric.id.startswith(f"D{i}.") for metric in METRICS) for i in range(1, 3)] == [11, 7]
qwen_source = (Path(__file__).parent / "video_eval" / "qwen.py").read_text()
assert 'MODEL = "Qwen3-VL-32B-Instruct"' in qwen_source
assert "ALLOWED_DEPLOYMENTS" in qwen_source

print("registry=ok metrics=18 distribution=11,7")
