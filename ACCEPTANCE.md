# 迁移验收结果

验收时间：2026-11-04T17:51:43.774170+08:00。结论：通过。

目标：`/pfs/pfs-spMCeu/eval_team/Video-eval/LT/eval_demension_1_2`。
原目录仅做读取；28个原代码/配置文件的SHA256、大小和修改时间与迁移前一致。

| 实测范围 | 请求指标数 | 有评分指标数 | 状态分布 |
|---|---:|---:|---|
| D1 | 11 | 9 | {"SUCCESS": 9, "UNCALIBRATED": 1, "NOT_APPLICABLE": 1} |
| D2 | 7 | 7 | {"UNCALIBRATED": 4, "SUCCESS": 2, "NEEDS_REVIEW": 1} |
| D1.2_targeted | 1 | 1 | {"SUCCESS": 1} |
| D1.4_targeted | 1 | 0 | {"UNCALIBRATED": 1} |

以上全部调用真实视频、专项算法和Qwen裁判；无 ERROR、MODEL_UNAVAILABLE 或报告错误字段。
D1通过JSON清单批量入口运行，D2运行全部7项；另外用数量/空间约束样例验证D1.2和D1.4，确保不是漏项或跳过。

D1.4保留原算法的原始空间关系准确率与检查证据，是否提供1–5分由原映射规则决定；未为凑齐分数改变公式。
UNCALIBRATED、NEEDS_REVIEW等成熟度/复核标记继承原项目，验收通过表示推理和输出链路正常，不等同于评分方法已人工标定。

运行环境和模型校验通过（20份权重校验记录），17项测试通过，38份项目Python文件语法检查通过。

## 结果位置

- D1: `results/migration_smoke_20261104/validated/batch_D1/D1/seedance-2-0-mini/010101/010101.json`
- D2: `results/migration_smoke_20261104/D2/020101.json`
- D1.2_targeted: `results/migration_smoke_20261104/supplemental_fixed/D1.2/010201.json`
- D1.4_targeted: `results/migration_smoke_20261104/validated/spatial_D1.4/010401.json`

完整机器可读验收：`migration/acceptance.json`。依赖/源代码复核：`migration/runtime_validation.json`、`migration/final_source_checks.json`。
迁移改动差异：`migration/source_diff.patch`。原版和中间尝试结果保留供诊断，正式结论以上述最终路径为准。

仍复用同机 LT/models/.venv 的Torch/CUDA库、8011端口Qwen裁判和共享ffmpeg；不存在指向LYX原项目的运行时链接。
