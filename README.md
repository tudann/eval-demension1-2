# 第 1、2 维度视频评估

迁移日期：2026-11-04。源项目为 `LYX/evaluation-tool/6_dimensions`；本目录使用独立代码和模型副本，默认仅评估 D1（11 项）与 D2（7 项）。共享模块保留原实现，未重新定义评分规则。

## 快速开始

```bash
cd /pfs/pfs-spMCeu/eval_team/Video-eval/LT/eval_demension_1_2
source env.sh
python selfcheck.py
python scripts/verify_runtime.py
python -m pytest -q
```

先运行 `nvidia-smi` 选择显存充足且未忙碌的 GPU。下列 `6` 是本次验收使用的卡号，后续须按实际负载调整。脚本不会停止其他任务，也不会自动启动裁判。

## 单视频

```bash
CUDA_VISIBLE_DEVICES=6 ./run_eval.sh /absolute/video.mp4 \
  --prompt '该视频实际使用的生成提示词' \
  --dimension 1 --output-dir results/my_d1

CUDA_VISIBLE_DEVICES=6 ./run_eval.sh /absolute/video.mp4 \
  --dimension 2 --output-dir results/my_d2
```

省略 `--dimension` 时运行 D1+D2 共18项；可重复传入该参数。也可使用 `--metric D1.1` 指定指标（显式指标优先于维度）。D1 必须提供 Prompt，否则返回证据不足。默认抽帧 2 fps、最多48帧，可通过 `--sample-fps` 和 `--max-frames` 调整。D2 的 MUSIQ、美学及人体结构官方算法内部另有自己的取帧方式，CLI 的抽帧参数不改变这些官方算法。

原风格入口可用：`python 01_instruction_following.py VIDEO --prompt PROMPT`、`python 02_visual_quality.py VIDEO`；使用前先 `source env.sh`。

输出为 JSON、CSV、Markdown，包含每项指标的原始证据、分数、状态和错误。出现 `ERROR`、`MODEL_UNAVAILABLE` 或错误字段时入口返回非零退出码。`UNCALIBRATED`、`NEEDS_REVIEW`、`NOT_APPLICABLE`、`INSUFFICIENT_EVIDENCE` 含义延续原项目，不代表全部都获得确定分数；原项目的算法分数尚未通过人工标签标定，迁移不会改变其成熟度。

## 批量评估

`inputs/manifest_d1_d2.json` 是原已生成视频清单中筛选出的35条 D1/D2 数据；视频仍指向 SHJ 的共享素材目录，原始 Prompt、参考图等元数据保留。`inputs/smoke.json` 和 `inputs/smoke/` 保存两条已复制到本目录的验收样本。

```bash
CUDA_VISIBLE_DEVICES=6 ./run_batch.sh inputs/manifest_d1_d2.json \
  --dimension 1 --dimension 2 --output-dir results/my_batch
# 接续已有输出
CUDA_VISIBLE_DEVICES=6 ./run_batch.sh inputs/manifest_d1_d2.json \
  --output-dir results/my_batch --resume
# 保留已成功输出，仅重跑失败报告
CUDA_VISIBLE_DEVICES=6 ./run_batch.sh inputs/manifest_d1_d2.json \
  --output-dir results/my_batch --retry-errors
```

清单支持 JSON 数组或 CSV，必填 `video_path`，可选 `prompt`、`dimension`（1/2）、`case_id`、`model`、`reference_images`。相对输入路径按清单所在目录解析；CSV参考图以分号分隔。每行有 `dimension` 时只评该维度，CLI维度参数用于筛选；无该字段时使用CLI选择或默认两维。每个视频写入独立的 `D维度/模型/Case` 子目录，汇总在 `batch_summary.json`。`--limit N` 可先少量试跑。默认重算；复用输出时应确保输入和参数未变，避免将 `--resume` 的历史报告误当作新推理。不要让多个批次同时写同一输出目录。

## 环境与依赖

- 主 Python：本目录 `.venv/bin/python`（Python 3.12）。大部分依赖为副本，Torch/CUDA 沿用服务器已有 `LT/models/.venv` 只读共享运行库，**不要在共享库内升级或卸载包**。
- D2.3/D2.6：本目录 `envs/vbench2`（Python 3.10 / Torch2.1 / MMCV2.1）独立子进程，解释器本体也复制在 `.python/`。人体结构和姿态默认CPU，避免旧CUDA内核与 B20Z 不兼容。
- 模型权重在本目录 `models/`，第三方实现位于 `third_party/`，均为复制文件；未链接到 LYX 原项目。
- 复用已有 Qwen3-VL-32B-Instruct 裁判：`http://127.0.0.1:8011/v1`。可在 source 前后设置 `QWEN3_VL_ENDPOINT` / `QWEN3_VL_MODEL`；裁判服务必须已运行，模型名必须是获准的 Qwen3-VL-32B-Instruct。运行需要足够GPU显存及可访问的裁判服务。
- ffmpeg 使用服务器已部署路径 `Video Evaluation Platform/runtimes/ffmpeg/bin`。这是同机迁移，并非完全打包的跨机器独立发行版。
- `requirements*.txt` 是原基础依赖说明，不足以完整重建专项环境；本次使用已验证的环境副本。完整版本清单保存在 `migration/`。

## 验收与追溯

最终验收已通过，见 `ACCEPTANCE.md`（含实际状态分布和结果路径）。

- 指标名称和后端：`METRICS.md`。
- 注册表及路由测试：`migration/unit_tests.log`。
- 环境、模型SHA256、符号链接和裁判：`migration/runtime_validation.json`；`python scripts/verify_runtime.py --check-source` 额外只读核对原代码。
- 真实推理：`results/migration_smoke_20261104/`，含D1/D2的报告、逐指标日志、裁判应答审计和 `acceptance.json`。
- 补充计数/空间实测：`results/migration_smoke_20261104/supplemental_fixed/`；原版静默漏项的诊断保留在 `supplemental/` 和 `migration/parsed_*.json`。
- 最终解析修复后的完整D1批量重测：`results/migration_smoke_20261104/validated/batch_D1/`，空间专项见同目录下的 `spatial_D1.4/`。最终汇总以 `migration/acceptance.json` 为准。
- 原代码前后哈希与修改时间：`migration/source_before.json` / `source_after.json`。
- 复制范围：`migration/copy_complete.json`。未迁移旧结果目录、无关维度入口及D4专用15GB LLaVA权重。

验收脚本 `source env.sh; EVAL_GPU=6 python scripts/smoke_d1_d2.py` 使用本目录两条真实视频并调用真实算法和裁判。再次验收时先选择空闲卡，并为新的测试修改脚本中的输出目录，避免覆盖本次记录。验收证明迁移后推理链路可运行，不代表重新验证了原评分方法的科学有效性。

## 副本内的兼容性修复

1. 将人体结构子环境、第三方旧模型链接及控制脚本路径迁到本目录，修复第三方仓库中含换行的损坏链接。
2. 注册表及CLI仅开放D1/D2；错误指标返回非零退出码，避免把缺少模型当成功。
3. 原解析器只用自由JSON，并静默丢弃未知类型。实测把“八个苹果”放在subject/attribute里，将空间关系标为非枚举类型，造成D1.2/D1.4漏项。副本改用严格JSON Schema、显式count/spatial示例和输出校验，失败显式报错并保存裁判原始应答。
4. 计数、空间关系的检测标签包含约束原子的目标实体，使道具不会因不在主角列表而漏检。保留原始算法公式与评分映射；本次未重新设计指标。

主环境仍复用 LT 的共享 Torch/CUDA 运行库；Qwen、ffmpeg也是本服务器共享服务/运行时。原 LYX 项目没有被更改。

## Git 版本管理

仓库为 `git@github.com:tudann/eval-demension1-2.git`。源码、配置、测试、迁移记录和保留本地修复的第三方源码纳入管理；模型、运行环境、视频和评估输出继续保存在服务器。提交范围、上游版本及日常操作见 [VERSION_CONTROL.md](VERSION_CONTROL.md)。
