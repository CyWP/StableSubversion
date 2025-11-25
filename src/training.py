import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import json

from accelerate import Accelerator
from accelerate.utils import tqdm
from hashlib import md5
from pathlib import Path, mkdir
from PIL import Image
from torch.utils.data import DataLoader, random_split
from typing import Dict, Tuple, List

from .config import Config
from .dataset import PromptDataset
from .model import StableSubversionPipeline
from .utils import CLIPModule

class Trainer:

    def __init__(self, config:Config):
        self.config = config
        self.create_stats_dir()

    def create_stats_dir(self):
        config = self.config
        self.run_dir = Path(config.run_dir) / config.run_name
        self.img_dir = self.run_dir / "img"
        mkdir(self.run_dir, parents=True)
        self.stats_file = self.run_dir / "stats.jsonl"
        with open(self.stats_file, "w") as f:
            f.write("")

    def log_stats(self, epoch:int, stats: Dict):
        with open(self.stats_file, "a") as f:
            f.write(json.dumps({epoch: stats}))

    def log_images(self, epoch: int, imgs:torch.Tensor, stats:List[Dict]):
        epoch_dir = self.img_dir / str(epoch)
        if not epoch_dir.is_dir():
            mkdir(epoch_dir, parents=True)
        imgs_reshaped = (imgs.detach().permute(0, 2, 3, 1).cpu().numpy()+1)*0.5
        for img, stat in zip(imgs_reshaped, stats):
            filename = md5(stats["prompt"].encode("utf-8")).hexdigest()
            img_path = epoch_dir / f"{filename}.png"
            stats_path = epoch_dir / f"{filename}.json"
            Image.fromarray(img).save(img_path)
            with open(stats_path, "w") as f:
                f.write(json.dump(stat))


    def train_lora(self):
        config = self.config
        accelerator = Accelerator(mixed_precision=config["mixed_precision"])
        device = accelerator.device
        self.device=device

        pipe = StableSubversionPipeline(
            model_name=config.model_name,
            torch_dtype=(
                torch.float16 if config.mixed_precision == "fp16" else torch.float32
            ),
        ).to(device)
        self.pipe = pipe

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
        clip.clip.eval()
        self. clip_scaler = transforms.Resize((224, 224))

        full_size = config.train_size + config.val_size + config.test_size
        dataset = PromptDataset(size=full_size, seed=config.generator)
        train_data, test_data = random_split(
            dataset,
            [config.train_size, config.test_size],
            generator=config.generator,
        )

        train_loader, test_loader = (
            DataLoader(
                split,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=True,
                generator=config.generator,
            )
            for split in [train_data, test_data]
        )

        pipe, optimizer, train_loader = accelerator.prepare(pipe, optimizer, train_loader)

        # Precompute embeddings
        with torch.no_grad():
            self.target_clip_emb = clip.encode([target_prompt], device)

        # TRAINING
        accelerator.print(">> Starting training...")
        epoch_bar = tqdm(range(config.epochs), desc="Epochs")
        for epoch in epoch_bar:
            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch} Batch")
            for batch in batch_bar:
                with accelerator.accumulate(pipe):
                    total_loss, distillation_loss, target_prompt_loss, adversarial_loss = self.train_step(batch)

                # Backprop / optimizer
                optimizer.zero_grad()
                accelerator.backward(total_loss)
                optimizer.step()

                # Logging
                batch_bar.set_postfix({
                    "loss": total_loss.item(),
                    "distill": distillation_loss.item(),
                    "target": target_prompt_loss.item(),
                    "adv": adversarial_loss.item()
                })
            

        accelerator.print(">> Training done.")
        return pipe

    def train_step(self, batch)->Tuple[torch.Tensor]:
        config = self.config
        device = self.device
        pipe = self.pipe
        prompts = batch["prompt"]
        with torch.no_grad():
            text_embeds = pipe.embed_text(prompts)
        num_steps = torch.randint(config.min_steps, config.max_steps+1, (1,)).item()
        # denoise at all timesteps
        lora_latents, ts = pipe.get_all_step_results(text_embeds, num_steps=num_steps)
        # disable lora
        distillation_loss = torch.tensor(0.0, device=device)
        with pipe.lora_disabled():
            for latents, t in zip(lora_latents, ts):
                with torch.no_grad():
                    noised = pipe.add_noise(latents, t)
                    lat = pipe.denoise_step(noised, t, text_embeds, 1.0)
                distillation_loss += F.mse_loss(lat, latents)
        # embed first denoised results in clip
        target_prompt_loss = torch.tensor(0.0, device=device)
        adversarial_loss = torch.tensor(0.0, device=device)
        for latents in lora_latents:
            imgs = pipe.decode_latents(latents)
            imgs_resized = self.clip_scaler(imgs)
            clip_adv_embs = self.clip.encode_image(imgs_resized, device)
            clip_prompt_embs = self.clip.encode_text(prompts, device)
            target_prompt_loss += (1-F.cosine_similarity(clip_adv_embs, self.target_clip_emb))
            adversarial_loss += (1+F.cosine_similarity(clip_adv_embs, clip_prompt_embs))
        # OPtimize
        # Average over timesteps
        distillation_loss /= len(ts)
        target_prompt_loss /= len(lora_latents)
        adversarial_loss /= len(lora_latents)

        total_loss = (config.distillation_weight * distillation_loss +
                        config.target_weight * target_prompt_loss +
                        config.adversarial_weight * adversarial_loss)
        return total_loss, distillation_loss, target_prompt_loss, adversarial_loss