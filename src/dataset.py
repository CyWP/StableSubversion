import torch

from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Union

from .utils import EasyDict


class PromptDataset(Dataset):
    """
    Just imagenetclasses
    """

    def __init__(
        self,
        txt_file: str = "./imagenet_categories.txt",
        size: Union[int, float] = 1000,
        seed: Union[int, torch.Generator] = 42,
    ):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.prompts = [line.strip() for line in lines]
        self.max_len = len(self.prompts)

        # Determine subset size
        if isinstance(size, float):
            size = int(size * self.max_len)
        if size > self.max_len:
            raise ValueError(f"Requested size {size} > dataset length {self.max_len}")

        # Generate subset indices with seed
        if isinstance(seed, int):
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = seed
        self.indices = torch.randperm(self.max_len, generator=generator)[:size]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Handle cases where idx is a tensor (e.g., from random_split)
        real_idx = self.indices[idx].item()
        return EasyDict(prompt=self.prompts[real_idx])


# class PromptDataset(Dataset):
#     """
#     PyTorch Dataset wrapping a HuggingFace text dataset.

#     Supports selecting a random subset with a fixed seed.
#     Fully compatible with `random_split` and DataLoader.
#     """

#     def __init__(
#         self,
#         subset: str = "2m_text_only",
#         size: Union[int, float] = 50000,
#         seed: Union[int, torch.Generator] = 42,
#     ):
#         # Load full HF dataset
#         self.ds = load_dataset("poloclub/diffusiondb", subset)["train"]
#         self.max_len = len(self.ds)

#         # Determine subset size
#         if isinstance(size, float):
#             size = int(size * self.max_len)
#         if size > self.max_len:
#             raise ValueError(f"Requested size {size} > dataset length {self.max_len}")

#         # Generate subset indices with seed
#         if isinstance(seed, int):
#             generator = torch.Generator().manual_seed(seed)
#         else:
#             generator = seed
#         self.indices = torch.randperm(self.max_len, generator=generator)[:size]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         # Handle cases where idx is a tensor (e.g., from random_split)
#         real_idx = self.indices[idx].item()
#         return EasyDict(prompt=self.ds[real_idx]["prompt"])
