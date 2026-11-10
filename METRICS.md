# D1/D2 指标与实现

本表由当前注册表生成。算法实现、原始证据及评分成熟度继承原项目；解析契约修复见 migration/source_diff.patch。

| ID | 指标 | 类型 | 后端 | 原有成熟度 |
|---|---|---|---|---|
| D1.1 | 主体正确性 | COMPOSED | grounded_sam2 | UNCALIBRATED |
| D1.2 | 主体数量准确性 | DIRECT | t2v_numeracy | UNCALIBRATED |
| D1.3 | 属性绑定准确性 | HYBRID | grounded_sam2+qwen | NEEDS_VALIDATION |
| D1.4 | 空间关系理解 | COMPOSED | grounded_sam2+video_depth | UNCALIBRATED |
| D1.5 | 动作指令遵循 | HYBRID | rtmpose+qwen | NEEDS_VALIDATION |
| D1.6 | 动作顺序遵循 | HYBRID | grounded_sam2+qwen | NEEDS_VALIDATION |
| D1.7 | 场景指令遵循 | QWEN | qwen | NEEDS_VALIDATION |
| D1.8 | 风格指令遵循 | HYBRID | vbench_appearance_style+qwen | NEEDS_VALIDATION |
| D1.9 | 摄影语言遵循 | HYBRID | cotracker3+qwen | NEEDS_VALIDATION |
| D1.10 | 否定约束遵循 | HYBRID | grounded_sam2+qwen | NEEDS_VALIDATION |
| D1.11 | 长 Prompt 完整度 | HYBRID | grounded_sam2+qwen | NEEDS_VALIDATION |
| D2.1 | 清晰度 | DIRECT | vbench_imaging_quality | UNCALIBRATED |
| D2.2 | 细节丰富度 | COMPOSED | opencv | UNCALIBRATED |
| D2.3 | 人体结构质量 | DIRECT | vbench2_human_anatomy | UNCALIBRATED |
| D2.4 | 材质表现 | QWEN | qwen | NEEDS_VALIDATION |
| D2.5 | 光照质量 | HYBRID | opencv+qwen | UNCALIBRATED |
| D2.6 | 视觉伪影 | HYBRID | vbench2_human_anatomy+qwen | NEEDS_VALIDATION |
| D2.7 | 美学质量 | DIRECT | vbench_aesthetic | UNCALIBRATED |

D1必须提供原始生成Prompt。D2可不提供Prompt。

UNCALIBRATED表示原方法尚未按人工标签校准；NEEDS_VALIDATION表示方法本身仍需验证。
NOT_APPLICABLE/INSUFFICIENT_EVIDENCE必须结合Prompt原文、解析审计和检测证据解释，不能当作模型质量的零分。

部分算法仅保留原始质量值，缺少人工标定时不强制编造1–5分。
