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
from .utils import CLIPModule, StatsDict, ImgTransform


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
        self, imgs: Union[torch.Tensor, List[Image.Image]], stats: List[StatsDict]
    ):
        epoch_dir = self.img_dir / str(self.current_epoch)
        if not epoch_dir.is_dir():
            Path.mkdir(epoch_dir, parents=True, exist_ok=True)
        if isinstance(imgs, torch.Tensor):
            imgs = ImgTransform.tensor2pil(imgs)
        for img, stat in zip(imgs, stats):
            filename = md5(stat.prompt.encode("utf-8")).hexdigest()
            img_path = epoch_dir / f"{filename}.png"
            stats_path = epoch_dir / f"{filename}.json"
            img.save(img_path)
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
            use_safetensors=False,
        ).to(device)
        self.pipe = pipe
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.create_lora(
            rank=config.lora_rank,
        )
        # Create and load LoRA for the target concept
        target_prompt = config.target_prompt

        optimizer = torch.optim.Adam(pipe.get_lora_params(), lr=config.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.sched_cycle)
        self.clip = CLIPModule(device, config.clip_name)
        self.clip.clip.eval()

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

        # Precompute embeddings
        with torch.no_grad():
            self.target_clip_emb = self.clip.encode_text([target_prompt])

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
            pipe.inspect_lora_weights()
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
        config = self.config
        device = self.device
        pipe = self.pipe
        accelerator = self.accelerator
        prompts = batch.prompt
        with torch.no_grad():
            text_embeds = pipe.embed_text(prompts)
        B = text_embeds.shape[0]
        num_steps = torch.randint(config.min_steps, config.max_steps, (1,)).item()
        loss_factor = num_steps * self.config.batch_size
        step_interval = 1000 // num_steps
        pipe.scheduler.set_timesteps(num_steps)
        latent = pipe.get_initial_noise(B)
        loss_dict = StatsDict(total=0, distillation=0, adversarial=0, target=0)
        for t in pipe.scheduler.timesteps:
            target_clip_emb = self.target_clip_emb.clone()
            latent = latent.detach().requires_grad_(True)
            t = min(t, torch.tensor(1000 - step_interval - 1))
            latent = pipe.denoise_step(latent, t, text_embeds, 1.0)
            with torch.no_grad(), pipe.lora_disabled():
                noised = pipe.add_noise(latent, t)
                denoised = pipe.denoise_step(noised, t, text_embeds, 1.0)
            distillation_loss = F.mse_loss(latent, denoised) / loss_factor

            imgs = pipe.decode_latents(latent)
            clip_adv_embs = self.clip.encode_image(imgs)
            clip_prompt_embs = self.clip.encode_text(prompts)
            target_prompt_loss = (
                1 - F.cosine_similarity(clip_adv_embs, target_clip_emb).mean()
            ) / loss_factor
            adversarial_loss = (
                1 + F.cosine_similarity(clip_adv_embs, clip_prompt_embs).mean()
            ) / loss_factor
            total_loss = (
                distillation_loss * config.distillation_weight
                + target_prompt_loss * config.target_weight
                + adversarial_loss * config.adversarial_weight
            )
            accelerator.backward(total_loss, retain_graph=True)
            loss_dict.accumulate(
                StatsDict(
                    total=total_loss,
                    distillation=distillation_loss,
                    adversarial=adversarial_loss,
                    target=target_prompt_loss,
                ).itemize()
            )

        return loss_dict

    @torch.no_grad()
    def test_step(self, batch) -> StatsDict:
        prompt = batch["prompt"]
        B = len(prompt)
        inference_steps = self.config.test_inference_steps
        cfg = self.config.test_cfg
        pipe = self.pipe
        imgs = self.pipe.generate(
            prompt, num_inference_steps=inference_steps, guidance_scale=cfg
        )["images"]
        imgs_tensor = ImgTransform.pil2tensor(imgs).to(self.device)
        clip_img_emb = self.clip.encode_image(imgs_tensor)
        clip_prompt_emb = self.clip.encode_text(prompt)
        target_prompt_losses = [
            1
            - F.cosine_similarity(
                clip_img_emb[i].unsqueeze(0), self.target_clip_emb
            ).mean()
            for i in range(B)
        ]
        adversarial_losses = [
            1
            + F.cosine_similarity(clip_img_emb[i].unsqueeze(0), clip_prompt_emb).mean()
            for i in range(B)
        ]
        loss_dict = StatsDict(
            target=target_prompt_losses, adversarial=adversarial_losses
        )
        gen_stats = [
            StatsDict(prompt=p, steps=inference_steps, cfg=cfg, target=t, adversarial=a)
            for p, t, a in zip(prompt, target_prompt_losses, adversarial_losses)
        ]
        self.log_images(imgs, gen_stats)
        target_loss = torch.tensor(target_prompt_losses).mean()
        adversarial_loss = torch.tensor(adversarial_losses).mean()
        return StatsDict(
            target_test=target_loss.item(), adversarial_test=adversarial_loss.item()
        )
