import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel
from typing import List

class CLIPModule(nn.Module):
    def __init__(self, name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(name)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)

    def encode_text(self, text_list: List[str], device: torch.device):
        tokens = self.tokenizer(text_list, padding=True, truncation=True,
                                return_tensors="pt").to(device)
        out = self.clip.get_text_features(**tokens)
        return out / out.norm(dim=-1, keepdim=True)

    def encode_image(self, images: torch.Tensor, device: torch.device):
        images = images.to(device)
        out = self.clip.get_image_features(images)
        return out / out.norm(dim=-1, keepdim=True)
