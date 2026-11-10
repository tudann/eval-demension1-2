from __future__ import annotations

import argparse
import json

from video_eval import EvaluationRequest, evaluate
from video_eval.report import write_reports


def run_dimension(dimension: int) -> int:
    parser = argparse.ArgumentParser(description=f"Evaluate dimension D{dimension}")
    parser.add_argument("video")
    parser.add_argument("--prompt")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=48)
    args = parser.parse_args()
    report = evaluate(EvaluationRequest(args.video, args.prompt, args.reference,
                                        enabled_dimensions=[dimension], sample_fps=args.sample_fps, max_frames=args.max_frames))
    paths = write_reports(report, args.output_dir)
    print(json.dumps({"summary": report["summary"], "reports": paths}, ensure_ascii=False, indent=2))
    return 2 if any(m["status"] in ("ERROR", "MODEL_UNAVAILABLE") or m.get("error") for m in report["metrics"]) else 0
