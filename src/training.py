import torch
import torch.nn.functional as F
import json

from accelerate import Accelerator
from hashlib import md5
from pathlib import Path
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Tuple, List, Union

from .config import Config
from .dataset import PromptDataset
from .model import StableSubversionPipeline
from .utils import EasyDict, StatsDict, ImgTransform

import logging

logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


class Trainer:

    def __init__(self, config: Config):
        self.config = config
        self.create_stats_dir()
        self.has_target = config.target_prompt is not None
        self.best_model_path = self.run_dir / "best.ckpt"
        self.last_model_path = self.run_dir / "last.ckpt"

    def create_stats_dir(self):
        config = self.config
        self.run_dir = Path(config.run_dir) / config.run_name
        self.img_dir = self.run_dir / "img"
        Path.mkdir(self.run_dir, parents=True, exist_ok=True)
        self.stats_file = self.run_dir / "stats.jsonl"
        with open(self.stats_file, "w") as f:
            f.write("")

    def log_stats(self, data: StatsDict):
        with open(self.stats_file, "a") as f:
            f.write(json.dumps({str(self.current_epoch): data}))
            f.write("\n")

    def log_images(
        self,
        base: Union[List[Image.Image], torch.Tensor],
        lora: Union[List[Image.Image], torch.Tensor],
        stats: list[StatsDict],
    ):
        epoch_dir = self.img_dir / str(self.current_epoch)
        if not epoch_dir.is_dir():
            Path.mkdir(epoch_dir, parents=True, exist_ok=True)
        if isinstance(lora, torch.Tensor):
            lora = ImgTransform.tensor2pil(lora)
        if isinstance(base, torch.Tensor):
            base = ImgTransform.tensor2pil(base)
        for img_base, img_lora, stat in zip(base, lora, stats):
            filename = "_".join(stat.prompt.split(" "))
            img_lora_path = epoch_dir / f"{filename}_lora.png"
            img_base_path = epoch_dir / f"{filename}_base.png"
            stats_path = epoch_dir / f"{filename}.json"
            img_lora.save(img_lora_path)
            img_base.save(img_base_path)
            with open(stats_path, "w") as f:
                f.write(json.dumps(stat.itemize()))

    def train_lora(self):
        config = self.config
        accelerator = Accelerator(mixed_precision=config["mixed_precision"])
        self.accelerator = accelerator
        device = accelerator.device
        self.device = device
        pipe = StableSubversionPipeline.from_pretrained(
            config.model_name,
            dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            use_auth_token=config.hf_auth_token or False,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.create_lora(
            rank=config.lora_rank,
        )
        # Create and load LoRA for the target concept
        target_prompt = config.target_prompt

        optimizer = torch.optim.Adam(pipe.get_lora_params(), lr=config.lr)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.sched_cycle, T_mult=2, eta_min=1e-6
        )
        full_size = config.train_size + config.test_size
        dataset = PromptDataset(size=full_size, seed=config.generator)
        train_data, test_data = random_split(
            dataset,
            [config.train_size, config.test_size],
            generator=config.generator,
        )

        train_loader, test_loader = (
            DataLoader(
                split,
                batch_size=config.minibatch_size,
                shuffle=True,
                drop_last=True,
                generator=config.generator,
            )
            for split in [train_data, test_data]
        )

        pipe, optimizer, train_loader = accelerator.prepare(
            pipe, optimizer, train_loader
        )

        # TRAINING
        accelerator.print(">> Starting training...")
        epoch_bar = tqdm(range(config.epochs), desc="Epochs")
        epoch_stats = StatsDict()
        best_loss = float("inf")
        for epoch in epoch_bar:
            self.current_epoch = epoch
            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch} Batch")
            loss_dict = StatsDict()
            optimizer.zero_grad()
            num_accumulated = 0
            for i, batch in enumerate(batch_bar):
                with accelerator.accumulate(self.pipe):
                    num_accumulated += config.minibatch_size
                    loss_dict = self.train_step(batch)
                    loss = loss_dict.total / config.batch_size
                    batch_bar.set_postfix(loss_dict.itemize())
                    epoch_stats.accumulate(loss_dict)
                    if num_accumulated >= config.batch_size or i + 1 == len(batch_bar):
                        num_accumulated = 0
                        optimizer.step()
                        optimizer.zero_grad()
            epoch_stats.divide(len(batch_bar))
            if epoch % config.test_interval == 0:
                test_bar = tqdm(test_loader, desc=f"Epoch {epoch} Test")
                test_stats = StatsDict()
                for batch in test_bar:
                    test_stats.accumulate(self.test_step(batch))
                test_stats.divide(len(test_bar))
                test_bar.set_postfix(test_stats.itemize())
                epoch_stats.accumulate(test_stats)
            epoch_bar.set_postfix(epoch_stats.itemize())
            scheduler.step()
            self.log_stats(epoch_stats)
            # save best and last model
            self.pipe.save_lora(self.last_model_path)
            if epoch_stats.total < best_loss:
                self.pipe.save_lora(self.best_model_path)
                best_loss = epoch_stats.total

        accelerator.print(">> Training done.")
        return pipe

    def train_step(self, batch) -> StatsDict:
        accelerator = self.accelerator
        config = self.config
        device = self.device
        pipe = self.pipe
        prompts = batch.prompt
        with torch.no_grad():
            text_embeds = pipe.embed_text(prompts)
        B = text_embeds.shape[0]
        num_steps = torch.randint(config.min_steps, config.max_steps, (1,)).item()
        loss_factor = num_steps * self.config.batch_size
        step_interval = 1000 // num_steps
        pipe.scheduler.set_timesteps(num_steps)
        latent = pipe.get_initial_noise(B)
        loss_dict = StatsDict(total=0, distillation=0, adversarial=0)
        for t in pipe.scheduler.timesteps:
            text_embeds = text_embeds.detach()
            t = min(t, torch.tensor(1000 - step_interval - 1)).item()
            lora_latent = pipe.denoise_step(latent, t, text_embeds, 1.0)
            with torch.no_grad(), pipe.lora_disabled():
                base_latent = pipe.denoise_step(latent, t, text_embeds, 1.0)
                renoised = pipe.add_noise(lora_latent, t)
                base_denoised = pipe.denoise_step(renoised, t, text_embeds, 1.0)
            distillation_loss = F.mse_loss(lora_latent, base_denoised) / loss_factor
            adversarial_loss = F.mse_loss(lora_latent, base_latent) / loss_factor
            total_loss = (
                config.distillation_weight * distillation_loss
                - config.adversarial_weight * adversarial_loss
            )
            accelerator.backward(total_loss, retain_graph=True)
            latent = lora_latent.detach()
            loss_dict.accumulate(
                StatsDict(
                    total=total_loss,
                    distillation=distillation_loss,
                    adversarial=adversarial_loss,
                ).itemize()
            )
        return loss_dict

    @torch.inference_mode()
    def test_step(self, batch) -> StatsDict:
        prompt = batch["prompt"]
        B = len(prompt)
        inference_steps = self.config.test_inference_steps
        cfg = self.config.test_cfg
        pipe = self.pipe
        latent = pipe.get_initial_noise(B)
        with pipe.lora_disabled():
            imgs_base = self.pipe.generate_from_latent(
                prompt,
                latents=latent,
                num_inference_steps=inference_steps,
                guidance_scale=cfg,
            )
        imgs_lora = self.pipe.generate_from_latent(
            prompt,
            latents=latent,
            num_inference_steps=inference_steps,
            guidance_scale=cfg,
        )
        gen_stats = [
            StatsDict(cfg=cfg, steps=inference_steps, prompt=p) for p in prompt
        ]
        self.log_images(imgs_base, imgs_lora, gen_stats)
        return StatsDict()
