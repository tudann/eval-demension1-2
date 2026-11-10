from __future__ import annotations

import importlib.util
import importlib
import json
import sys
from pathlib import Path


def patch_nms_for_new_gpu() -> None:
    """Use TorchVision NMS when the MMCV wheel lacks this GPU's kernel image."""
    import torch
    from torchvision.ops import nms as torchvision_nms

    nms_module = importlib.import_module("mmcv.ops.nms")

    def compatible_nms(boxes, scores, iou_threshold, offset=0,
                       score_threshold=0, max_num=-1):
        if offset not in (0, 1):
            raise ValueError("offset must be 0 or 1")
        valid = scores > score_threshold if score_threshold > 0 else None
        original_indices = torch.arange(scores.numel(), device=scores.device)
        if valid is not None:
            boxes = boxes[valid]
            scores = scores[valid]
            original_indices = original_indices[valid]
        keep_local = torchvision_nms(boxes, scores, iou_threshold)
        if max_num > 0:
            keep_local = keep_local[:max_num]
        dets = torch.cat((boxes[keep_local], scores[keep_local, None]), dim=1)
        keep = original_indices[keep_local]
        return dets, keep

    nms_module.nms = compatible_nms


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "human_anatomy":
        raise SystemExit("usage: vbench2_worker.py human_anatomy VIDEO")
    patch_nms_for_new_gpu()
    adapter_path = Path(__file__).with_name("vbench2_adapter.py")
    spec = importlib.util.spec_from_file_location("isolated_vbench2_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {adapter_path}")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    result = adapter.human_anatomy(sys.argv[2])
    print("__VBENCH2_RESULT__=" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
