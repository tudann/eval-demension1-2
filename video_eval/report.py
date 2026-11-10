from __future__ import annotations

import csv
import json
from pathlib import Path


def write_reports(report: dict, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(report["video"]["path"]).stem
    json_path, csv_path, md_path = output / f"{stem}.json", output / f"{stem}.csv", output / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dimension_id", "name", "status", "applicable", "score",
                                                        "confidence", "implementation_status", "method", "reason", "error"])
        writer.writeheader()
        for metric in report["metrics"]:
            writer.writerow({key: metric.get(key) for key in writer.fieldnames})
    lines = [f"# Evaluation: {stem}", "", f"- Video: `{report['video']['path']}`",
             f"- Scored: {report['summary']['scored']}/{report['summary']['requested']}", "",
             "| ID | Metric | Status | Score | Implementation |", "|---|---|---:|---:|---|"]
    for metric in report["metrics"]:
        lines.append(f"| {metric['dimension_id']} | {metric['name']} | {metric['status']} | "
                     f"{metric['score'] if metric['score'] is not None else '—'} | {metric['implementation_status']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
