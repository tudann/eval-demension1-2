"""Place the D1/D2 manifest into dimension/mode/case directories.

The scored prompt is written to prompt_final.txt. The video and reference
images stay the original files and are linked into each case directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "inputs" / "manifest_d1_d2.json"
DEST = ROOT / "inputs" / "cases"
DIMENSION_DIRS = {1: "01_instruction_following", 2: "02_visual_quality"}
MODES = {"t2v": "t2va", "i2v": "f2va", "ref2v": "r2va"}


def link(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink() or destination.exists():
        if destination.exists() and destination.resolve() == source and destination.stat().st_ino == source.stat().st_ino:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source)


def stage() -> None:
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in rows:
        mode = MODES[row["task_type"]]
        case_dir = DEST / DIMENSION_DIRS[int(row["dimension"])] / mode / row["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt = row.get("prompt") or ""
        prompt_path = case_dir / "prompt_final.txt"
        if prompt:
            prompt_path.write_text(prompt, encoding="utf-8")
        elif prompt_path.exists():
            prompt_path.unlink()
        link(Path(row["video_path"]), case_dir / "video.mp4")
        for index, ref in enumerate(row.get("reference_images") or [], start=1):
            source = Path(ref)
            link(source, case_dir / f"input_{index:02d}{source.suffix.lower()}")
        metadata = {
            "case_id": row["case_id"],
            "dimension": int(row["dimension"]),
            "mode": mode,
            "model": row.get("model") or "default",
            "task_type": row.get("task_type"),
            "original_prompt": row.get("original_prompt") or "",
            "sha256": row.get("sha256"),
            "source_video_path": row["video_path"],
            "source_reference_images": row.get("reference_images") or [],
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "root": str(DEST)}, ensure_ascii=False))


if __name__ == "__main__":
    stage()
