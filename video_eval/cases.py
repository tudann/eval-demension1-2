"""Discover D1/D2 cases laid out like the dimension 13–20 directories."""
from __future__ import annotations

import json
from pathlib import Path

DIMENSION_DIRS = {
    "01_instruction_following": 1,
    "02_visual_quality": 2,
}
MODES = ("t2va", "f2va", "fl2va", "r2va")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _metadata(case_dir: Path) -> dict:
    path = case_dir / "metadata.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metadata must be an object: {path}")
    return data


def _images(case_dir: Path) -> list[Path]:
    found = []
    for index in range(1, 9):
        stem = f"input_{index:02d}"
        found.extend(sorted(path for path in case_dir.iterdir()
                            if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS))
    return found


def _reference_images(case_dir: Path, mode: str) -> list[str]:
    images = _images(case_dir)
    if mode == "f2va":
        images = images[:1]
    elif mode == "fl2va":
        images = images[:2]
    elif mode == "r2va":
        images = images[:8]
    else:
        images = []
    return [str(path.absolute()) for path in images]


def _cases_under(dimension_dir: Path, dimension: int) -> list[dict]:
    rows = []
    if not dimension_dir.is_dir():
        return rows
    for mode_dir in sorted(path for path in dimension_dir.iterdir() if path.is_dir()):
        if mode_dir.name not in MODES:
            continue
        for case_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
            video = case_dir / "video.mp4"
            prompt_path = case_dir / "prompt_final.txt"
            meta = _metadata(case_dir)
            row = {
                "case_id": case_dir.name,
                "dimension": dimension,
                "mode": mode_dir.name,
                "model": meta.get("model") or "default",
                "video_path": str(video.absolute()) if video.is_file() else str(video),
                "prompt": prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else None,
                "reference_images": _reference_images(case_dir, mode_dir.name),
            }
            if meta.get("original_prompt"):
                row["original_prompt"] = meta["original_prompt"]
            rows.append(row)
    return rows


def discover_cases(root: Path) -> list[dict]:
    """Read ``<dimension>/<mode>/<case_id>/{prompt_final.txt,video.mp4}``."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if root.name in DIMENSION_DIRS:
        return _cases_under(root, DIMENSION_DIRS[root.name])
    rows = []
    for name, dimension in DIMENSION_DIRS.items():
        rows.extend(_cases_under(root / name, dimension))
    return rows
