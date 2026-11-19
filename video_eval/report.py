"""Write D1/D2 results in the dimension 13–20 case directory layout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_eval.cases import DIMENSION_DIRS, MODES

RUN_NAME = "local"
DIMENSION_KEYS = {number: name for name, number in DIMENSION_DIRS.items()}
BAD_STATUS = {"ERROR", "MODEL_UNAVAILABLE"}


def case_directory(output_dir: str | Path, dimension: int, mode: str, case_id: str, run_name: str = RUN_NAME) -> Path:
    return Path(output_dir) / "runs" / run_name / DIMENSION_KEYS[dimension] / mode / case_id


def result_file(output_dir: str | Path, dimension: int, mode: str, case_id: str, run_name: str = RUN_NAME) -> Path:
    return case_directory(output_dir, dimension, mode, case_id, run_name) / "result.json"


def placement_from_video(video_path: str) -> tuple[str, str, int | None]:
    """Return mode, case id, and dimension when the video lives in a case directory."""
    parts = Path(video_path).absolute().parts
    for index, name in enumerate(parts[:-1]):
        if name in DIMENSION_DIRS and index + 2 < len(parts) and parts[index + 1] in MODES:
            return parts[index + 1], parts[index + 2], DIMENSION_DIRS[name]
    return "direct", Path(video_path).stem or "video", None


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _scores(metrics: list[dict[str, Any]]) -> list[float]:
    return [float(item["score"]) for item in metrics if isinstance(item.get("score"), (int, float))]


def _status(metrics: list[dict[str, Any]]) -> str:
    errors = [item for item in metrics if item.get("status") in BAD_STATUS or item.get("error")]
    scores = _scores(metrics)
    if errors and scores:
        return "incomplete_subpoints"
    if errors:
        return "error"
    if scores:
        return "ok"
    return "no_score"


def _result(report: dict[str, Any], metrics: list[dict[str, Any]], *, dimension: int, mode: str,
            case_id: str, model: str | None, video_path: str | None, reference_images: list[str]) -> dict[str, Any]:
    scores = _scores(metrics)
    lowest_name, lowest_score = None, None
    for item in metrics:
        if not isinstance(item.get("score"), (int, float)):
            continue
        if lowest_score is None or float(item["score"]) < lowest_score:
            lowest_name, lowest_score = item.get("dimension_id"), float(item["score"])
    video = report.get("video") or {}
    subpoints = {}
    for item in metrics:
        subpoints[item.get("dimension_id") or item.get("name") or "metric"] = {
            "name": item.get("name"),
            "status": item.get("status"),
            "applicable": item.get("applicable"),
            "score": item.get("score"),
            "reason": item.get("reason"),
            "error": item.get("error"),
            "implementation_status": item.get("implementation_status"),
            "method": item.get("method"),
        }
    return {
        "case_id": f"{mode}/{case_id}",
        "source_case_id": case_id,
        "input_id": mode,
        "dimension": DIMENSION_KEYS[dimension],
        "status": _status(metrics),
        "score": round(sum(scores) / len(scores), 3) if scores else None,
        "score_rule": "mean",
        "lowest_subpoint": lowest_name,
        "lowest_score": lowest_score,
        "subpoints": subpoints,
        "summary": {"requested": len(metrics), "scored": len(scores),
                    "status_counts": {str(status): sum(item.get("status") == status for item in metrics)
                                      for status in {item.get("status") for item in metrics}}},
        "metrics": metrics,
        "case": {
            "case_id": case_id,
            "mode": mode,
            "model": model or "default",
            "video_path": video_path or video.get("path"),
            "reference_images": reference_images,
            "width": video.get("width"),
            "height": video.get("height"),
            "duration": video.get("duration"),
        },
    }


def _write_case(directory: Path, result: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    facts = {item.get("dimension_id"): item.get("raw_metrics") or {} for item in metrics}
    checklist = {
        "version": "d1_d2_metrics_v1",
        "dimension": result["dimension"],
        "case_id": result["source_case_id"],
        "input_id": result["input_id"],
        "items": [{"id": item.get("dimension_id"), "name": item.get("name"), "method": item.get("method"),
                   "implementation_status": item.get("implementation_status")} for item in metrics],
    }
    answers = {
        "status": result["status"],
        "answers": [{"id": item.get("dimension_id"), "status": item.get("status"), "score": item.get("score"),
                     "applicable": item.get("applicable"), "reason": item.get("reason"), "error": item.get("error")}
                    for item in metrics],
    }
    lines = [f"# {result['case_id']}", "", f"- 维度: {result['dimension']}", f"- 维度分: {result['score']}",
             f"- 状态: {result['status']}", "", "| 指标 | 名称 | 状态 | 分数 |", "|---|---|---|---:|"]
    for item in metrics:
        score = "—" if item.get("score") is None else item.get("score")
        lines.append(f"| {item.get('dimension_id')} | {item.get('name')} | {item.get('status')} | {score} |")
    lines.append("")
    _json(directory / "facts.json", facts)
    _json(directory / "checklist.json", checklist)
    _json(directory / "answers.json", answers)
    _json(directory / "result.json", result)
    (directory / "stage_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_case_outputs(report: dict[str, Any], output_dir: str | Path, *, dimensions: list[int], mode: str,
                       case_id: str, model: str | None = None, video_path: str | None = None,
                       reference_images: list[str] | None = None, run_name: str = RUN_NAME) -> list[dict[str, Any]]:
    """Write one case directory for each evaluated dimension."""
    refs = reference_images or []
    results = []
    for dimension in dimensions:
        prefix = f"D{dimension}."
        metrics = [item for item in report.get("metrics") or [] if str(item.get("dimension_id", "")).startswith(prefix)]
        result = _result(report, metrics, dimension=dimension, mode=mode, case_id=case_id, model=model,
                         video_path=video_path, reference_images=refs)
        _write_case(case_directory(output_dir, dimension, mode, case_id, run_name), result, metrics)
        results.append(result)
    return results


def write_unscored(output_dir: str | Path, *, dimension: int, mode: str, case_id: str, reason: str,
                   model: str | None = None, video_path: str | None = None, run_name: str = RUN_NAME) -> dict[str, Any]:
    result = {
        "case_id": f"{mode}/{case_id}",
        "source_case_id": case_id,
        "input_id": mode,
        "dimension": DIMENSION_KEYS[dimension],
        "status": "skipped_evaluate",
        "score": None,
        "score_rule": "mean",
        "reason": reason,
        "subpoints": {},
        "summary": {"requested": 0, "scored": 0, "status_counts": {}},
        "metrics": [],
        "case": {"case_id": case_id, "mode": mode, "model": model or "default", "video_path": video_path,
                 "reference_images": []},
    }
    _write_case(case_directory(output_dir, dimension, mode, case_id, run_name), result, [])
    return result


def write_run_summary(output_dir: str | Path, results: list[dict[str, Any]], run_name: str = RUN_NAME) -> dict[str, Any]:
    base = Path(output_dir) / "runs" / run_name
    scores = [float(item["score"]) for item in results if isinstance(item.get("score"), (int, float))]
    per_dimension: dict[str, dict[str, Any]] = {}
    for name in sorted({item.get("dimension") for item in results}):
        rows = [item for item in results if item.get("dimension") == name]
        row_scores = [float(item["score"]) for item in rows if isinstance(item.get("score"), (int, float))]
        per_dimension[str(name)] = {
            "n": len(rows),
            "scored": len(row_scores),
            "mean_score": round(sum(row_scores) / len(row_scores), 3) if row_scores else None,
        }
    per_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({str((item.get("case") or {}).get("mode") or item.get("input_id") or "unknown") for item in results}):
        rows = [item for item in results if str((item.get("case") or {}).get("mode") or item.get("input_id")) == mode]
        row_scores = [float(item["score"]) for item in rows if isinstance(item.get("score"), (int, float))]
        per_mode[mode] = {
            "n": len(rows),
            "scored": len(row_scores),
            "skipped": sum(item.get("score") is None for item in rows),
            "mean_score": round(sum(row_scores) / len(row_scores), 3) if row_scores else None,
        }
    summary = {
        "run_name": run_name,
        "n": len(results),
        "scored": len(scores),
        "skipped": sum(item.get("score") is None for item in results),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "per_dimension": per_dimension,
        "per_mode": per_mode,
    }
    _json(base / "summary.json", summary)
    (base / "per_case.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8")
    dimension_lines = "".join(
        f"- {name}: {row['scored']}/{row['n']} scored, mean {row['mean_score']} / 5\n"
        for name, row in per_dimension.items()
    )
    mode_lines = "".join(
        f"- {mode}: {row['scored']}/{row['n']} scored, skipped {row['skipped']}, mean {row['mean_score']} / 5\n"
        for mode, row in per_mode.items()
    )
    (base / "summary.md").write_text(
        f"# D1-D2\n\n- Cases: {summary['n']}\n- Scored: {summary['scored']}\n"
        f"- Skipped: {summary['skipped']}\n- Mean score: {summary['mean_score']} / 5\n\n"
        f"## By dimension\n\n{dimension_lines}\n## By input mode\n\n{mode_lines}",
        encoding="utf-8",
    )
    return summary


def write_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write a single-video report into the case directory layout."""
    request = report.get("request") or {}
    video_path = (report.get("video") or {}).get("path") or request.get("video_path") or "video.mp4"
    mode, case_id, path_dimension = placement_from_video(str(video_path))
    metadata = request.get("metadata") or {}
    dimensions = list(request.get("enabled_dimensions") or [])
    if not dimensions:
        dimensions = sorted({int(item["dimension_id"][1]) for item in report.get("metrics") or []
                             if str(item.get("dimension_id", "")).startswith("D") and item["dimension_id"][1].isdigit()})
    if not dimensions:
        dimensions = [path_dimension] if path_dimension else [1]
    results = write_case_outputs(
        report, output_dir, dimensions=dimensions, mode=mode, case_id=case_id,
        model=metadata.get("model"), video_path=str(video_path),
        reference_images=list(request.get("reference_images") or []),
    )
    write_run_summary(output_dir, results)
    return {item["dimension"]: str(result_file(output_dir, DIMENSION_DIRS[item["dimension"]], item["input_id"], item["source_case_id"]))
            for item in results}
