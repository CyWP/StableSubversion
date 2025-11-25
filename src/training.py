import torch
from accelerate import Accelerator
from accelerate.utils import tqdm
from torch import nn
from torch.utils.data import DataLoader, random_split
from typing import Dict, List

from .config import Config
from .dataset import PromptDataset
from .model import StableSubversionPipeline
from .utils import CLIPModule


def train_lora(config: Config):

    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    pipe = StableSubversionPipeline(
        model_name=config.model_name,
        torch_dtype=(
            torch.float16 if config.mixed_precision == "fp16" else torch.float32
        ),
    ).to(device)

    # Create and load LoRA for the target concept
    target_prompt = config.target_prompt
    lora_params = pipe.create_lora(
        name=target_prompt,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        train=True,
    )
    pipe.set_lora(target_prompt)
    optimizer = torch.optim.Adam(lora_params, lr=config.lr)

    clip = CLIPModule(config.clip_name).to(device)

    full_size = config.train_size + config.val_size + config.test_size
    dataset = PromptDataset(size=full_size, seed=config.generator)
    train_data, val_data, test_data = random_split(
        dataset,
        [config.train_size, config.val_size, config.test_size],
        generator=config.generator,
    )

    train_loader, val_loader, test_loader = (
        DataLoader(
            split,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            generator=config.generator,
        )
        for split in [train_data, val_data, test_data]
    )

    pipe, optimizer, train_loader = accelerator.prepare(pipe, optimizer, train_loader)

    # Precompute embeddings
    with torch.no_grad():
        target_text_emb = pipe.embed_text([target_prompt]).to(device)
        target_clip_emb = clip.encode([target_prompt], device)

    # TRAINING
    accelerator.print(">> Starting training...")
    epoch_bar = tqdm(range(config.epochs), desc="Epochs")
    for epoch in epoch_bar:
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch} Batch")
        for batch in tqdm(train_loader, desc="Batch"):
            with accelerator.accumulate(pipe):
                prompts = batch["prompt"]
                # gen timesteps
                # gen noise
                # denoise at all timesteps
                # disable lora
                # renoise generated images
                # denoise with lora disabled
                # embed first denoised results in clip
                # MSE with clip embedding of target
                # -MSE with clip embedding of prompts
                # MSE between sets of denoised images
                # OPtimize

    accelerator.print(">> Training done.")
    return pipe
