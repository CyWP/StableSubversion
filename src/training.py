import torch
import torch.nn.functional as F
import json

from accelerate import Accelerator
from hashlib import md5
from pathlib import Path
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Tuple, List, Union

from .config import Config
from .dataset import PromptDataset
from .model import StableSubversionPipeline, LoRAEMA
from .utils import EasyDict, StatsDict, ImgTransform

import logging

logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


class Trainer:

    def __init__(self, config: Config):
        self.config = config
        self.create_stats_dir()
        self.best_model_path = self.run_dir / "best.ckpt"
        self.last_model_path = self.run_dir / "last.ckpt"

    def create_stats_dir(self):
        config = self.config
        self.run_dir = Path(config.run_dir) / config.run_name
        self.img_dir = self.run_dir / "img"
        self.ckpt_dir = self.run_dir / "checkpoints"
        Path.mkdir(self.run_dir, parents=True, exist_ok=True)
        Path.mkdir(self.img_dir, parents=True, exist_ok=True)
        Path.mkdir(self.ckpt_dir, parents=True, exist_ok=True)
        self.cfg_file = self.run_dir / "config.json"
        with open(self.cfg_file, "w") as f:
            f.write(config.json())
        self.stats_file = self.run_dir / "stats.jsonl"
        with open(self.stats_file, "w") as f:
            f.write("")

    def log_stats(self, data: StatsDict):
        with open(self.stats_file, "a") as f:
            f.write(json.dumps({str(self.current_epoch): data}))
            f.write("\n")

    def log_images(
        self,
        imgs: Union[torch.Tensor, List[Image.Image]],
        stats: List[StatsDict],
        subfolder: str = None,
        suffix: str = None,
        base: Union[torch.Tensor, List[Image.Image]] = None,
    ):
        if subfolder is None:
            subfolder = str(self.current_epoch)
        imgs_dir = self.img_dir / str(subfolder)
        if not imgs_dir.is_dir():
            Path.mkdir(imgs_dir, parents=True, exist_ok=True)
        if isinstance(imgs, torch.Tensor):
            imgs = ImgTransform.tensor2pil(imgs)
        if base is not None and isinstance(base, torch.Tensor):
            base = ImgTransform.tensor2pil(base)
        for i in range(len(imgs)):
            img = imgs[i]
            stat = stats[i]
            filename = "_".join(stat.prompt.split(" "))
            suf = stat.cfg if suffix is None else suffix
            img_path = imgs_dir / f"{filename}_{suf}.png"
            stats_path = imgs_dir / f"{filename}_{suf}.json"
            img.save(img_path)
            if base:
                b = base[i]
                b_path = imgs_dir / f"{filename}_{suf}_base.png"
                b.save(b_path)
            with open(stats_path, "w") as f:
                f.write(json.dumps(stat.itemize()))

    def train_lora(self):
        config = self.config
        self.target = config.target_prompt
        self.use_target = self.target is not None
        self.use_negative_prompt = config.use_negative_prompt
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
        self.ema = LoRAEMA(pipe, decay=config.ema_decay)
        ema = self.ema
        self.test_latents = pipe.get_initial_noise(config.minibatch_size)

        optimizer = torch.optim.AdamW(pipe.get_lora_params(), lr=config.lr)
        scheduler = StepLR(
            optimizer, step_size=config.sched_step_size, gamma=config.sched_gamma
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
        if self.use_target:
            with torch.no_grad():
                self.target_embedding = pipe.embed_text(config.target_prompt)
        # TRAINING
        accelerator.print(">> Generating baseline test images...")
        accelerator.print(">> Starting training...")
        epoch_bar = tqdm(range(config.epochs), desc="Epochs")
        epoch_stats = StatsDict()
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
                        ema.update()
                        optimizer.zero_grad()
            epoch_stats.divide(len(batch_bar))
            if epoch % config.test_interval == 0:
                test_bar = tqdm(test_loader, desc=f"Epoch {epoch} Test")
                test_stats = StatsDict()
                with ema.applied():
                    for batch in test_bar:
                        test_stats.accumulate(self.test_step(batch))
                test_stats.divide(len(test_bar))
                test_bar.set_postfix(test_stats.itemize())
                epoch_stats.accumulate(test_stats)
            epoch_bar.set_postfix(epoch_stats.itemize())
            scheduler.step()
            self.log_stats(epoch_stats)
            # save ema and raw model for epoch
            ema.save(self.ckpt_dir / f"model_{epoch}_ema.ckpt")
            pipe.save_lora(self.ckpt_dir / f"model_{epoch}_train.ckpt")

        accelerator.print(">> Training done.")
        return pipe

    def train_step(self, batch) -> StatsDict:
        accelerator = self.accelerator
        config = self.config
        device = self.device
        pipe = self.pipe
        prompts = batch.prompt
        B = len(prompts)
        with torch.no_grad():
            text_embeds = pipe.embed_text(prompts)
            if self.use_target and self.use_negative_prompt:
                self.target_embedding = pipe.embed_text([self.target] * B, prompts)
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
                base_latent = pipe.denoise_step(
                    latent,
                    t,
                    self.target_embedding if self.use_target else text_embeds,
                    1.0,
                )
                renoised = pipe.add_noise(lora_latent, t)
                base_denoised = pipe.denoise_step(renoised, t, text_embeds, 1.0)
            distillation_loss = F.mse_loss(lora_latent, base_denoised) / loss_factor
            adversarial_loss = F.mse_loss(lora_latent, base_latent) / loss_factor
            adv_mult = 1.0 if self.use_target else -1.0
            total_loss = (
                config.distillation_weight * distillation_loss
                + config.adversarial_weight * adversarial_loss * adv_mult
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
    def test_step(self, batch, subfolder: str = None, suffix=None) -> StatsDict:
        prompt = batch["prompt"]
        B = len(prompt)
        inference_steps = self.config.test_inference_steps
        pipe = self.pipe
        latent = self.test_latents[:B].clone()

        if len(latent.shape) < len(self.test_latents.shape):
            latent = latent.unsqueeze(0)
        for cfg in self.config.test_cfgs:
            with pipe.lora_disabled():
                imgs_base = self.pipe.generate_from_latent(
                    prompt,
                    latents=latent,
                    num_inference_steps=inference_steps,
                    guidance_scale=cfg,
                )
            imgs = self.pipe.generate_from_latent(
                prompt,
                latents=latent,
                num_inference_steps=inference_steps,
                guidance_scale=cfg,
            )
            gen_stats = [
                StatsDict(cfg=cfg, steps=inference_steps, prompt=p) for p in prompt
            ]
            self.log_images(imgs, gen_stats, subfolder, suffix, base=imgs_base)
        return StatsDict()
