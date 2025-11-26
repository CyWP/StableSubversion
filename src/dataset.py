import torch

from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Union, Dict

from .utils import EasyDict


class PromptDataset(Dataset):

    def __init__(
        self,
        subset: str = "2m_random_1k",
        size: Union[int, float] = 50000,
        seed: Union[int | torch.Generator] = 42,
    ):
        self.ds = load_dataset("poloclub/diffusiondb", subset)["train"]
        max_len = len(self.ds)
        if isinstance(size, float):
            size = int(size * max_len)

        if size < len(self.ds):
            raise ValueError(
                f"Requested subset of length {size} is too large for dataset of length {max_len}"
            )
        if isinstance(seed, int):
            seed = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(max_len, generator=seed)[:size]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> EasyDict:
        real_idx = self.indices[idx]
        prompt = self.ds[real_idx]["prompt"]
        return EasyDict(prompt=prompt)
