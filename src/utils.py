import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPModel
from typing import List


class CLIPModule(nn.Module):
    def __init__(self, name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(name)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)

    @torch.no_grad()
    def encode(self, text_list: List[str], device: torch.device):
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt").to(device)
        out = self.clip.get_text_features(**tokens)
        return out / out.norm(dim=-1, keepdim=True)
