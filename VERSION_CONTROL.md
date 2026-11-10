# Git 版本管理

仓库：`git@github.com:tudann/eval-demension1-2.git`，主分支：`main`。

## 提交范围

纳入评估入口、D1/D2 实现、配置与标定参数、测试、运行/验收脚本、输入 JSON 清单、迁移记录，以及 `third_party/` 中保留本地兼容性修复的第三方源码和许可证。
第三方代码按普通文件保存，上游地址、提交版本和导入时的本地修改见 `third_party/UPSTREAM_SOURCES.json`；克隆本仓库无需初始化子模块。服务器上原有第三方仓库的 `.git` 元数据保留在本机，不提交到此仓库。

`.gitignore` 排除了模型权重、Python/专项环境、解释器副本、缓存、备份、二进制扩展、视频样本、动画演示、运行输出和本地密钥。已有文件仍保留在服务器上。
`models/` 的权重清单见 `environment_manifest.json`，环境版本见 `migration/*requirements.freeze.txt`。`inputs/` 中的 JSON 清单纳入管理，实际视频不上传。

## 在现有服务器目录提交修改

```bash
cd /pfs/pfs-spMCeu/eval_team/Video-eval/LT/eval_demension_1_2
git status --short
git diff
git add video_eval tests scripts  # 按实际修改选择路径
git commit -m "Describe the evaluation change"
git push origin main
```

查看已忽略的运行资源：`git status --short --ignored`。
不要用 `git add -f` 强制提交已排除的模型、运行环境、视频或密钥。

## 从 GitHub 克隆

```bash
git clone git@github.com:tudann/eval-demension1-2.git
cd eval-demension1-2
```

克隆获得的是源码快照。运行前须按 `README.md` 配置依赖、模型、视频和 Qwen 裁判服务；这些资源继续由服务器保存。现有 `requirements*.txt` 不足以完整重建专项环境，环境版本和模型校验值可从上述清单追溯。
`ACCEPTANCE.md` 和 `migration/` 中引用的 `results/` 及验收视频路径对应服务器上的历史验收材料，实际输出和视频不随 Git 克隆提供。
