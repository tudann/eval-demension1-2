#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIDEO_EVAL_ROOT="$ROOT"
export VIDEO_EVAL_PYTHON="${VIDEO_EVAL_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT:$ROOT/.deps:$ROOT/third_party/VBench:$ROOT/third_party/VBench/VBench-2.0:$ROOT/third_party/Grounded-SAM-2:$ROOT/third_party/co-tracker:$ROOT/third_party/Video-Depth-Anything:$ROOT/third_party/DWPose/ControlNet-v1-1-nightly"
export QWEN3_VL_ENDPOINT="${QWEN3_VL_ENDPOINT:-http://127.0.0.1:8011/v1}"
export QWEN3_VL_MODEL="${QWEN3_VL_MODEL:-Qwen3-VL-32B-Instruct}"
export HF_HOME="$ROOT/models/huggingface"
export TORCH_HOME="$ROOT/models/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VIDEO_EVAL_ANATOMY_DEVICE="${VIDEO_EVAL_ANATOMY_DEVICE:-cuda}"
export VIDEO_EVAL_ONNX_DEVICE="${VIDEO_EVAL_ONNX_DEVICE:-cuda}"
_VIDEO_EVAL_CUDA_LIBS="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="${_VIDEO_EVAL_CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _VIDEO_EVAL_CUDA_LIBS
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VIDEO_EVAL_SIGLIP_FRAME_BATCH=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TZ=Asia/Shanghai
export PATH="/pfs/pfs-spMCeu/eval_team/Video-eval/Video Evaluation Platform/runtimes/ffmpeg/bin:$ROOT/.venv/bin:$PATH"
