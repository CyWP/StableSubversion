from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import numpy as np

from pathlib import Path
from PIL import Image
from transformers import CLIPTokenizer, CLIPModel
from typing import List, Dict, Union


class CLIPModule(nn.Module):
    def __init__(self, device, name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.scaler = transforms.Resize((224, 224))
        self.device = device
        self.to(device)

    def encode_text(self, text_list: List[str]):
        tokens = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.clip.get_text_features(**tokens)
        return out / out.norm(dim=-1, keepdim=True)

    def encode_image(self, images: torch.Tensor):
        out = self.clip.get_image_features(self.scaler(images))
        return out / out.norm(dim=-1, keepdim=True)


class EasyDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._convert_value(v)

    def _convert_value(self, v):
        # Convert dicts recursively
        if isinstance(v, dict):
            return EasyDict(v)
        # Convert dicts inside lists or tuples recursively
        elif isinstance(v, list):
            return [self._convert_value(x) for x in v]
        elif isinstance(v, tuple):
            return tuple(self._convert_value(x) for x in v)
        # leave everything else as is
        else:
            return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"'EasyDict' object has no attribute '{name}'") from e

    def __setattr__(self, name, value):
        self[name] = self._convert_value(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"'EasyDict' object has no attribute '{name}'") from e


class StatsDict(EasyDict):

    def itemize(self) -> StatsDict:
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.item()
        return self

    def accumulate(self, data: Dict) -> StatsDict:
        for k, v in data.items():
            if k in self:
                self[k] += v
            else:
                self[k] = v
        return self

    def divide(self, val: float) -> StatsDict:
        for k, v in self.items():
            if isinstance(v, (float, torch.Tensor)):
                self[k] = v / val
        return self

    def log(self, epoch: int, file: Path):
        with open(file, "a") as f:
            f.write(json.dumps({epoch: self}))


class ImgTransform:

    @staticmethod
    def tensor2pil(img: torch.Tensor) -> Image.Image:
        B, C, H, W = img.shape
        img_reshaped = ((img.unsqueeze(0) + 1) * 0.5).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(img_reshaped[i].unsqueeze(0)) for i in range(B)]

    @staticmethod
    def pil2tensor(img: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            return torch.tensor(np.asarray(img) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        return torch.stack(
            [torch.tensor(np.asarray(i) * 2 - 1).permute(2, 0, 1) for i in img], dim=0
        )
