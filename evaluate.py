from __future__ import annotations

import argparse
import json

from video_eval import EvaluationRequest, evaluate
from video_eval.report import write_reports


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Evaluate generated video on dimensions D1-D2")
    value.add_argument("video")
    value.add_argument("--prompt")
    value.add_argument("--reference", action="append", default=[])
    value.add_argument("--metric", action="append", dest="metrics")
    value.add_argument("--dimension", action="append", type=int, dest="dimensions", choices=(1, 2))
    value.add_argument("--output-dir", default="reports")
    value.add_argument("--sample-fps", type=float, default=2.0)
    value.add_argument("--max-frames", type=int, default=48)
    return value


def main() -> int:
    args = parser().parse_args()
    request = EvaluationRequest(args.video, args.prompt, args.reference, {}, args.metrics, args.dimensions,
                                args.sample_fps, args.max_frames)
    report = evaluate(request)
    paths = write_reports(report, args.output_dir)
    print(json.dumps({"summary": report["summary"], "reports": paths}, ensure_ascii=False, indent=2))
    return 2 if any(m["status"] in ("ERROR", "MODEL_UNAVAILABLE") or m.get("error") for m in report["metrics"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
