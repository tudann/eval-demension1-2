from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VBENCH = ROOT / "third_party" / "VBench"
if str(VBENCH) not in sys.path:
    sys.path.insert(0, str(VBENCH))

class _BoundedImageEncoder:
    """ViT encodes images independently; batching bounds peak attention memory."""
    def __init__(self,model,batch_size=8):
        self.model=model
        self.batch_size=batch_size
    def encode_image(self,images):
        import torch
        with torch.inference_mode():
            return torch.cat([self.model.encode_image(chunk) for chunk in images.split(self.batch_size)],dim=0)


def temporal_flickering(video_path: str) -> dict:
    """Official VBench Temporal Flickering implementation."""
    from vbench.temporal_flickering import cal_score
    return {"value": float(cal_score(video_path)), "source": "VBench.temporal_flickering.cal_score"}


@lru_cache(maxsize=1)
def _musiq(device: str):
    from pyiqa.archs.musiq_arch import MUSIQ
    weight = ROOT / "models" / "vbench" / "pyiqa" / "musiq_spaq_ckpt-358bb6af.pth"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    model = MUSIQ(pretrained_model_path=str(weight)).to(device)
    model.eval()
    return model


def imaging_quality(video_path: str, device: str = "cuda") -> dict:
    """Official VBench Imaging Quality (MUSIQ-SPAQ) implementation."""
    from vbench.imaging_quality import technical_quality
    value, details = technical_quality(_musiq(device), [video_path], device,
                                       imaging_quality_preprocessing_mode="longer")
    return {"value": float(value), "details": details, "source": "VBench.imaging_quality.technical_quality"}


@lru_cache(maxsize=1)
def _aesthetic_models(device: str):
    import clip
    from vbench.aesthetic_quality import get_aesthetic_model
    clip_path = ROOT / "models" / "vbench" / "clip" / "ViT-L-14.pt"
    aes_dir = ROOT / "models" / "vbench" / "aesthetic"
    clip_model, preprocess = clip.load(str(clip_path), device=device)
    return get_aesthetic_model(str(aes_dir)).to(device), clip_model, preprocess


def aesthetic_quality(video_path: str, device: str = "cuda") -> dict:
    """Official VBench/LAION aesthetic predictor."""
    from vbench.aesthetic_quality import laion_aesthetic
    aesthetic_model, clip_model, _ = _aesthetic_models(device)
    value, details = laion_aesthetic(aesthetic_model, clip_model, [video_path], device)
    return {"value": float(value), "details": details, "source": "VBench.aesthetic_quality.laion_aesthetic"}


@lru_cache(maxsize=1)
def _background_models(device: str):
    import clip
    path = ROOT / "models" / "vbench" / "clip" / "ViT-L-14.pt"
    return clip.load(str(path), device=device)


def background_consistency(video_path: str, device: str = "cuda") -> dict:
    """Official VBench Background Consistency implementation."""
    from vbench.background_consistency import background_consistency as official
    model, preprocess = _background_models(device)
    import torch
    with torch.inference_mode():
        value, details = official(_BoundedImageEncoder(model), preprocess, [video_path], device, False)
    return {"value": float(value), "details": details, "source": "VBench.background_consistency",
            "image_batch_size":8,"inference_mode":True}


@lru_cache(maxsize=1)
def _motion_model(device: str):
    from vbench.motion_smoothness import MotionSmoothness
    config = ROOT / "third_party" / "VBench" / "vbench" / "third_party" / "amt" / "cfgs" / "AMT-S.yaml"
    weight = ROOT / "models" / "vbench" / "amt" / "amt-s.pth"
    return MotionSmoothness(str(config), str(weight), device)


def motion_smoothness(video_path: str, device: str = "cuda") -> dict:
    """Official VBench AMT motion smoothness implementation."""
    value = _motion_model(device).motion_score(video_path)
    return {"value": float(value), "source": "VBench.motion_smoothness.MotionSmoothness"}


def appearance_style(video_path: str, style_text: str, device: str = "cuda") -> dict:
    """Official VBench Appearance Style frame/text CLIP similarity."""
    import clip
    from vbench.appearance_style import appearance_style as official
    import torch
    model, _ = _background_models(device)
    payload = [{"video_list": [video_path],
                "auxiliary_info": {"appearance_style": style_text}}]
    try:
        clip.tokenize([style_text])
    except RuntimeError as exc:
        if "context length" not in str(exc):
            raise
        return _long_appearance_style(model,video_path,style_text,device)
    with torch.inference_mode():
        value, details = official(model, payload, device)
    return {"value": float(value), "details": details,
            "query": style_text, "source": "VBench.appearance_style.appearance_style"}

def _style_token_chunks(text):
    from clip.simple_tokenizer import SimpleTokenizer
    import torch
    tokenizer=SimpleTokenizer()
    raw=tokenizer.encode(text)
    chunks=[raw[i:i+75] for i in range(0,len(raw),75)] or [[]]
    result=torch.zeros((len(chunks),77),dtype=torch.long)
    for i,chunk in enumerate(chunks):
        seq=[tokenizer.encoder["<|startoftext|>"],*chunk,tokenizer.encoder["<|endoftext|>"]]
        result[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
    return result,len(raw)

def _long_appearance_style(model,video_path,text,device):
    """All text tokens retained; long-query evidence uses mean chunk similarity."""
    import torch
    import numpy as np
    from PIL import Image
    from vbench.utils import load_video,clip_transform_Image
    tokens,token_count=_style_token_chunks(text)
    tokens=tokens.to(device)
    transform=clip_transform_Image(224)
    per_frame=[]
    per_chunk=[]
    with torch.inference_mode():
        for frame in load_video(video_path,return_tensor=False):
            image=transform(Image.fromarray(frame)).to(device).unsqueeze(0)
            _,logits=model(image,tokens)
            similarities=(logits[:,0]/100).float().cpu().numpy()
            per_frame.append(float(similarities.mean()))
            per_chunk.append(similarities.tolist())
    if not per_frame:
        raise ValueError("No frames available for appearance style")
    value=float(np.mean(per_frame))
    return {"value":value,"query":text,"source":"VBench CLIP appearance style with lossless long-text chunking",
            "text_token_count":token_count,"text_chunk_count":len(tokens),
            "text_aggregation":"mean_chunk_similarity","truncated":False,
            "details":[{"video_path":video_path,"video_results":value,"frame_results":per_frame,
                        "per_chunk_mean":np.mean(per_chunk,axis=0).tolist()}]}
