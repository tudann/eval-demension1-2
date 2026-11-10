from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricSpec:
    id: str
    name: str
    kind: str
    backend: str
    implementation_status: str
    prompt_required: bool = False


_ROWS = [
    ("D1.1", "主体正确性", "COMPOSED", "grounded_sam2", "UNCALIBRATED", 1),
    ("D1.2", "主体数量准确性", "DIRECT", "t2v_numeracy", "UNCALIBRATED", 1),
    ("D1.3", "属性绑定准确性", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 1),
    ("D1.4", "空间关系理解", "COMPOSED", "grounded_sam2+video_depth", "UNCALIBRATED", 1),
    ("D1.5", "动作指令遵循", "HYBRID", "rtmpose+qwen", "NEEDS_VALIDATION", 1),
    ("D1.6", "动作顺序遵循", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 1),
    ("D1.7", "场景指令遵循", "QWEN", "qwen", "NEEDS_VALIDATION", 1),
    ("D1.8", "风格指令遵循", "HYBRID", "vbench_appearance_style+qwen", "NEEDS_VALIDATION", 1),
    ("D1.9", "摄影语言遵循", "HYBRID", "cotracker3+qwen", "NEEDS_VALIDATION", 1),
    ("D1.10", "否定约束遵循", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 1),
    ("D1.11", "长 Prompt 完整度", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 1),
    ("D2.1", "清晰度", "DIRECT", "vbench_imaging_quality", "UNCALIBRATED", 0),
    ("D2.2", "细节丰富度", "COMPOSED", "opencv", "UNCALIBRATED", 0),
    ("D2.3", "人体结构质量", "DIRECT", "vbench2_human_anatomy", "UNCALIBRATED", 0),
    ("D2.4", "材质表现", "QWEN", "qwen", "NEEDS_VALIDATION", 0),
    ("D2.5", "光照质量", "HYBRID", "opencv+qwen", "UNCALIBRATED", 0),
    ("D2.6", "视觉伪影", "HYBRID", "vbench2_human_anatomy+qwen", "NEEDS_VALIDATION", 0),
    ("D2.7", "美学质量", "DIRECT", "vbench_aesthetic", "UNCALIBRATED", 0),
    ("D3.1", "帧间连续性", "COMPOSED", "vbench_motion_smoothness+cotracker3", "UNCALIBRATED", 0),
    ("D3.2", "闪烁控制", "DIRECT", "vbench_temporal_flickering", "UNCALIBRATED", 0),
    ("D3.3", "几何稳定性", "COMPOSED", "grounded_sam2+cotracker3", "UNCALIBRATED", 0),
    ("D3.4", "纹理稳定性", "COMPOSED", "opencv_flow", "UNCALIBRATED", 0),
    ("D3.5", "光照稳定性", "COMPOSED", "opencv", "UNCALIBRATED", 0),
    ("D3.6", "背景稳定性", "DIRECT", "vbench_background", "UNCALIBRATED", 0),
    ("D4.1", "身份一致性", "DIRECT", "vbench2_identity", "UNCALIBRATED", 0),
    ("D4.2", "五官一致性", "COMPOSED", "insightface", "UNCALIBRATED", 0),
    ("D4.3", "发型一致性", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 0),
    ("D4.4", "服装一致性", "DIRECT", "vbench2_human_clothes_llava_video", "UNCALIBRATED", 0),
    ("D4.5", "配饰一致性", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 0),
    ("D4.6", "身材一致性", "COMPOSED", "rtmpose+grounded_sam2", "UNCALIBRATED", 0),
    ("D4.7", "多视角一致性", "DIRECT", "vbench2_multiview", "UNCALIBRATED", 0),
    ("D4.8", "遮挡恢复一致性", "COMPOSED", "grounded_sam2+insightface", "UNCALIBRATED", 0),
    ("D5.1", "动作完整性", "HYBRID", "rtmpose+qwen", "NEEDS_VALIDATION", 1),
    ("D5.2", "动作流畅性", "DIRECT", "vbench_motion_smoothness", "UNCALIBRATED", 0),
    ("D5.3", "动作自然性", "HYBRID", "rtmpose+qwen", "NEEDS_VALIDATION", 0),
    ("D5.4", "姿态合理性", "COMPOSED", "rtmpose", "UNCALIBRATED", 0),
    ("D5.5", "速度合理性", "COMPOSED", "cotracker3", "UNCALIBRATED", 0),
    ("D5.6", "运动轨迹合理性", "COMPOSED", "cotracker3", "UNCALIBRATED", 0),
    ("D5.7", "起止状态正确性", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 1),
    ("D5.8", "高动态运动能力", "COMPOSED", "rtmpose+cotracker3", "UNCALIBRATED", 0),
    ("D6.1", "接触准确性", "COMPOSED", "grounded_sam2+rtmpose", "UNCALIBRATED", 0),
    ("D6.2", "抓握合理性", "HYBRID", "rtmpose+grounded_sam2+qwen", "NEEDS_VALIDATION", 0),
    ("D6.3", "拿取与放置", "COMPOSED", "grounded_sam2+cotracker3", "UNCALIBRATED", 0),
    ("D6.4", "开关操作", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 1),
    ("D6.5", "食物与饮品交互", "HYBRID", "grounded_sam2+qwen", "NEEDS_VALIDATION", 0),
    ("D6.6", "双人交互", "HYBRID", "rtmpose+grounded_sam2+qwen", "NEEDS_VALIDATION", 0),
    ("D6.7", "多主体交互", "HYBRID", "grounded_sam2+rtmpose+qwen", "NEEDS_VALIDATION", 0),
    ("D6.8", "交互后状态保持", "COMPOSED", "grounded_sam2+cotracker3", "UNCALIBRATED", 0),
]

METRICS = tuple(MetricSpec(a, b, c, d, e, bool(f)) for a, b, c, d, e, f in _ROWS if a.startswith(("D1.", "D2.")))
BY_ID = {metric.id: metric for metric in METRICS}


def select(enabled: list[str] | None = None, dimensions: list[int] | None = None) -> list[MetricSpec]:
    ids = set(enabled or [])
    dims = set(dimensions or [])
    if dims - {1, 2}:
        raise ValueError("Only dimensions 1 and 2 are available")
    unknown = ids - BY_ID.keys()
    if unknown:
        raise ValueError(f"unknown metric ids: {sorted(unknown)}")
    if ids:
        return [metric for metric in METRICS if metric.id in ids]
    if dims:
        return [metric for metric in METRICS if int(metric.id[1]) in dims]
    return list(METRICS)
