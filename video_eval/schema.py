from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RunStatus = Literal[
    "SUCCESS",
    "NOT_APPLICABLE",
    "MODEL_UNAVAILABLE",
    "DETECTION_FAILED",
    "INSUFFICIENT_EVIDENCE",
    "UNCALIBRATED",
    "NEEDS_REVIEW",
    "ERROR",
]


@dataclass(slots=True)
class Evidence:
    observation: str
    start: float | None = None
    end: float | None = None
    frame_index: int | None = None
    region: list[float] | None = None


@dataclass(slots=True)
class MetricResult:
    dimension_id: str
    name: str
    method: str
    implementation_status: str
    status: RunStatus
    applicable: bool | None
    score: float | None = None
    confidence: float | None = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    reason: str = ""
    needs_review: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
