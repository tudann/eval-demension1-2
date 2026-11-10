from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache

from .qwen import ALLOWED_DEPLOYMENTS, MODEL, QwenUnavailable

ATOM_TYPES = {"subject", "count", "attribute", "spatial", "action", "scene", "style", "camera", "negative"}


@lru_cache(maxsize=128)
def parse(prompt: str) -> dict:
    """Decompose a generation prompt with the only approved general VLM."""
    endpoint = os.getenv("QWEN3_VL_ENDPOINT", "http://127.0.0.1:8011/v1")
    model = os.getenv("QWEN3_VL_MODEL", MODEL)
    if model not in ALLOWED_DEPLOYMENTS:
        raise QwenUnavailable(f"refusing non-approved parser model: {model}")
    instruction = (
        "Decompose the video generation prompt into atomic constraints. Return strict JSON: "
        "{atoms:[{id,type,text,subjects,objects,attributes,target_count,relation,order,critical}], "
        "subject_labels:[string], object_labels:[string], has_people:boolean, has_interaction:boolean}. "
        f"type must be one of {sorted(ATOM_TYPES)}. Preserve explicit negative constraints. Prompt: {prompt}"
    )
    payload = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": instruction}],
               "response_format": {"type": "json_object"}}
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    api_key = os.getenv("QWEN3_VL_API_KEY")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    last_error: Exception | None = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.load(response)
            last_error = None
            break
        except TimeoutError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    parsed = json.loads(body["choices"][0]["message"]["content"])
    atoms = []
    for atom in parsed.get("atoms", []):
        # Audio-only constraints are outside dimensions 1-6.  Qwen sometimes
        # emits them despite the closed enum; ignore them instead of failing all
        # visual metrics for the video.
        if atom.get("type") not in ATOM_TYPES:
            continue
        index = len(atoms)
        atom.setdefault("id", f"A{index + 1}")
        atoms.append(atom)
    parsed["atoms"] = atoms
    parsed.setdefault("subject_labels", [])
    parsed.setdefault("object_labels", [])
    parsed["parser_model"] = model
    return parsed


METRIC_ATOMS = {
    "D1.1": {"subject"}, "D1.2": {"count"}, "D1.3": {"attribute"}, "D1.4": {"spatial"},
    "D1.5": {"action"}, "D1.6": {"action"}, "D1.7": {"scene"}, "D1.8": {"style"},
    "D1.9": {"camera"}, "D1.10": {"negative"},
    "D1.11": ATOM_TYPES,
}


def relevant(parsed: dict, metric_id: str) -> list[dict]:
    accepted = METRIC_ATOMS.get(metric_id)
    return [atom for atom in parsed.get("atoms", []) if accepted is None or atom.get("type") in accepted]
