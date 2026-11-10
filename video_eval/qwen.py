from __future__ import annotations

import base64
import json
import os
import math
import re
import time
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any

import cv2

from .features import VideoFeatures
from .rubrics import rubric

MODEL = "Qwen3-VL-32B-Instruct"
ALLOWED_DEPLOYMENTS = {MODEL}


class QwenUnavailable(RuntimeError):
    pass

def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value,bool):
        raise ValueError("boolean is not a timestamp")
    if isinstance(value,(int,float)):
        result=float(value)
    elif isinstance(value,str):
        text=value.strip().lower()
        text=re.sub(r"\s*(?:seconds?|secs?|s|秒)$","",text).strip()
        if not re.fullmatch(r"\d+(?::\d{1,2}){0,2}(?:\.\d+)?",text):
            raise ValueError(f"invalid timestamp: {value!r}")
        result=0.0
        for part in text.split(":"):
            result=result*60+float(part)
    else:
        raise ValueError(f"invalid timestamp type: {type(value).__name__}")
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result

def response_schema(metric_id: str, duration: float) -> dict:
    stamp={"type":["number","null"],"minimum":0,"maximum":float(duration)}
    return {"type":"json_schema","json_schema":{"name":"video_metric_judgment","strict":True,"schema":{
        "type":"object","additionalProperties":False,
        "properties":{
            "dimension_id":{"type":"string","const":metric_id},"applicable":{"type":"boolean"},
            "score":{"type":["integer","null"],"minimum":1,"maximum":5},
            "confidence":{"type":"number","minimum":0,"maximum":1},
            "evidence":{"type":"array","items":{"type":"object","additionalProperties":False,
                "properties":{"start":stamp,"end":stamp,"observation":{"type":"string","minLength":1}},
                "required":["start","end","observation"]}},
            "reason":{"type":"string"},"needs_review":{"type":"boolean"}},
        "required":["dimension_id","applicable","score","confidence","evidence","reason","needs_review"]}}}

def _audit_response(body: dict, metric_id: str, error: str | None) -> None:
    directory=os.getenv("VIDEO_EVAL_JUDGE_AUDIT_DIR")
    if directory:
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        (path/f"{metric_id}_{time.time_ns()}.json").write_text(json.dumps(
            {"response":body,"validation_error":error},ensure_ascii=False,indent=2))


def validate_answer(answer: dict[str, Any], metric_id: str, duration: float) -> dict[str, Any]:
    if answer.get("dimension_id") != metric_id:
        raise ValueError(f"judge returned dimension_id={answer.get('dimension_id')!r}, expected {metric_id}")
    if not isinstance(answer.get("applicable"), bool):
        raise ValueError("judge applicable must be boolean")
    score = answer.get("score")
    if answer["applicable"]:
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError("applicable judge result must contain integer score 1..5")
    elif score is not None:
        raise ValueError("non-applicable judge result must have null score")
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError("judge confidence must be in [0,1]")
    evidence = answer.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("judge evidence must be a list")
    normalized=[]
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError(f"judge evidence[{index}] must be an object")
        observation=item.get("observation")
        if not observation:
            observation=next((item[k] for k in ("description","text","event") if isinstance(item.get(k),str) and item[k].strip()),None)
        if not isinstance(observation,str) or not observation.strip():
            raise ValueError(f"judge evidence[{index}] lacks observation")
        item=dict(item)
        if "timestamp" in item and item.get("start") is None and item.get("end") is None:
            item["start"]=item["end"]=item["timestamp"]
        start, end = item.get("start"), item.get("end")
        for label, value in (("start", start), ("end", end)):
            if value is None:
                continue
            value=_seconds(value)
            if value < 0 or value > duration:
                raise ValueError(f"judge evidence[{index}].{label} is outside the video")
            item[label] = value
        start, end = item.get("start"), item.get("end")
        if start is not None and end is not None and start > end:
            raise ValueError(f"judge evidence[{index}] has reversed interval")
        cleaned={"observation":observation,"start":item.get("start"),"end":item.get("end")}
        if isinstance(item.get("frame_index"),int) and not isinstance(item["frame_index"],bool):
            cleaned["frame_index"]=item["frame_index"]
        if isinstance(item.get("region"),list) and len(item["region"])==4:
            cleaned["region"]=item["region"]
        normalized.append(cleaned)
    answer["evidence"]=normalized
    if answer["applicable"] and not evidence:
        answer["needs_review"] = True
    if answer["applicable"] and evidence and all(x.get("start") is None and x.get("end") is None for x in normalized):
        answer["needs_review"]=True
    answer.setdefault("needs_review", False)
    answer.setdefault("reason", "")
    return answer


def _data_url(frame) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("failed to encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _compact(value: Any, depth: int = 0) -> Any:
    """Drop frame-level dumps before they are sent to the judge."""
    if isinstance(value, dict):
        return {key: _compact(item, depth + 1) for key, item in value.items()
                if key not in {"frames", "step_displacements_norm", "timestamps", "counts"}}
    if isinstance(value, list):
        if len(value) > 6:
            return {"count": len(value), "head": [_compact(item, depth + 1) for item in value[:2]]}
        return [_compact(item, depth + 1) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, 4)
    return value


def judge(video: VideoFeatures, metric_id: str, name: str, prompt: str | None, evidence: dict[str, Any]) -> dict:
    endpoint = os.getenv("QWEN3_VL_ENDPOINT", "http://127.0.0.1:8011/v1")
    api_key = os.getenv("QWEN3_VL_API_KEY")
    configured_model = os.getenv("QWEN3_VL_MODEL", MODEL)
    if configured_model not in ALLOWED_DEPLOYMENTS:
        raise QwenUnavailable(f"refusing non-approved judge model: {configured_model}")
    picks = list(range(0, len(video.frames), max(1, len(video.frames) // 8)))[:8]
    content: list[dict] = [{"type": "text", "text": (
        f"Evaluate only {metric_id} {name}. Metric rubric: {rubric(metric_id)} "
        f"Generation prompt: {prompt or '[not provided]'}. "
        f"Objective evidence: {json.dumps(_compact(evidence), ensure_ascii=False, default=str)}. "
        "Use only visible evidence. Return strict JSON with keys dimension_id, applicable, score "
        "(integer 1-5 or null), confidence (0-1), evidence (list of start/end/observation), reason, needs_review. "
        "Evidence start/end must be numeric seconds within the video duration, or null if unknown. "
        "Every evidence item needs an observation string. Do not emit timestamp aliases or frame labels as times."
    )}]
    for index in picks:
        content.extend([{"type": "text", "text": f"timestamp={video.timestamps[index]:.3f}s"},
                        {"type": "image_url", "image_url": {"url": _data_url(video.frames[index])}}])
    max_tokens = int(os.getenv("QWEN3_VL_MAX_TOKENS", "1024"))
    if not 128 <= max_tokens <= 4096:
        raise ValueError("QWEN3_VL_MAX_TOKENS must be between 128 and 4096")
    payload = {"model": configured_model, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": content}],
               "response_format": response_schema(metric_id,video.duration)}
    last_error=None
    for attempt in range(2):
        request=urllib.request.Request(endpoint.rstrip("/")+"/chat/completions",
            data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
        if api_key:
            request.add_header("Authorization",f"Bearer {api_key}")
        body={}
        try:
            with urllib.request.urlopen(request,timeout=240) as response:
                body=json.load(response)
            answer=validate_answer(json.loads(body["choices"][0]["message"]["content"]),metric_id,video.duration)
            _audit_response(body,metric_id,None)
            return answer
        except urllib.error.HTTPError as exc:
            detail=exc.read().decode("utf-8","replace")[:600]
            _audit_response(body,metric_id,f"HTTP {exc.code}: {detail}")
            raise RuntimeError(f"HTTP Error {exc.code}: {detail}") from exc
        except (ValueError,TypeError,KeyError,TimeoutError) as exc:
            last_error=exc
            _audit_response(body,metric_id,f"{type(exc).__name__}: {exc}")
            if attempt==0:
                payload["messages"][0]["content"][0]["text"]+=f" Previous response validation failed: {exc}. Return valid evidence without inventing observations or scores."
    raise last_error
