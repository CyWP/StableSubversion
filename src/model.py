import torch
import torch.nn.functional as F
import inspect

from contextlib import contextmanager
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    AttnProcessor,
    LoRAAttnProcessor2_0,
)
from pathlib import Path
from typing import List, Tuple, Optional


class StableSubversionPipeline(
    StableDiffusionPipeline,
):

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker: bool = True,
    ):
        sig = inspect.signature(super().__init__)
        params = sig.parameters
        if "image_encoder" in params:
            super().__init__(
                vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                image_encoder,
                requires_safety_checker,
            )
        else:
            super().__init__(
                vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                requires_safety_checker,
            )

        for param in self.unet.parameters():
            param.requires_grad_(False)
        for param in self.vae.parameters():
            param.requires_grad_(False)
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        self.img0_dict = dict()
        self.img1_dict = dict()
        self._lora_layers = {}
        self._current_lora = None

    def create_lora(self, name: str, rank: int = 8):
        if name in self._lora_layers:
            raise ValueError(f"LoRA '{name}' already exists.")
        unet = self.unet

        # initialize UNet LoRA
        unet_lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                raise NotImplementedError(
                    "name must start with up_blocks, mid_blocks, or down_blocks"
                )
            unet_lora_attn_procs[name] = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            ).to(self.device)
        self._lora_layers[name] = unet_lora_attn_procs
        unet.set_attn_processor(unet_lora_attn_procs)
        unet_lora_layers = AttnProcsLayers(unet.attn_processors)

        return [p for p in unet_lora_layers.parameters() if p.requires_grad]

    def save_lora(self, path: Path, name: Optional[str] = None):
        name = self._current_lora if name is None else name
        if name not in self._lora_layers:
            raise ValueError(f"LoRA '{name}' doesn't exist.")

        # gather all processor state dicts
        state = {k: v.state_dict() for k, v in self._lora_layers[name].items()}
        torch.save(state, path)

    def load_lora(self, name: str, path: str):
        loaded_state = torch.load(path, map_location=self.device)

        unet_lora_attn_procs = {}
        for attn_name, st in loaded_state.items():
            # Recreate a LoRA processor with the correct sizes
            hidden_size = st["lora_down.weight"].shape[1]  # typical for LoRA
            cross_attention_dim = st.get("lora_up.weight", None)
            cross_attention_dim = (
                cross_attention_dim.shape[0]
                if cross_attention_dim is not None
                else None
            )

            proc = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=st["lora_up.weight"].shape[1],
            )
            proc.load_state_dict(st)
            unet_lora_attn_procs[attn_name] = proc

        self._lora_layers[name] = unet_lora_attn_procs
        self.unet.set_attn_processor(unet_lora_attn_procs)
        self._current_lora = name

        lora_params = [
            p
            for p in AttnProcsLayers(self.unet.attn_processors).parameters()
            if p.requires_grad
        ]
        return lora_params

    def enable_lora(self, name: str):
        if name == self._current_lora:
            return
        if name not in self._lora_layers:
            raise ValueError(f"LoRA '{name}' not found")
        self.unet.set_attn_processor(self._lora_layers[name])
        self._current_lora = name

    def disable_lora(self):
        default_procs = {}
        for name, _ in self.unet.attn_processors.items():
            default_procs[name] = AttnProcessor()
        self.unet.set_attn_processor(default_procs)
        self._current_lora = None

    @contextmanager
    def lora_disabled(self):
        prev = self._current_lora
        self.disable_lora()
        try:
            yield self
        finally:
            if prev is not None:
                self.enable_lora(prev)

    def embed_text(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # if negative prompt is a single string -> replicate it for whole batch
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            else:
                assert (
                    len(negative_prompt) == batch_size
                ), "negative_prompt list must be same length as prompt list"

        tok_pos = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        pos_embeds = self.text_encoder(tok_pos.input_ids)[0]

        if negative_prompt is None:
            return pos_embeds  # (B, seq, dim)

        tok_neg = self.tokenizer(
            negative_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        neg_embeds = self.text_encoder(tok_neg.input_ids)[0]

        # Classifier-free guidance format: [neg, pos]
        return torch.cat([neg_embeds, pos_embeds], dim=0)

    def encode_image(self, image):
        return self.vae.encode(image).latent_dist.sample() * 0.18215

    def decode_latents(self, latents):
        latents = latents / 0.18215
        return self.vae.decode(latents).sample

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(x0.device)
        a = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sigma = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return a * x0 + sigma * noise

    def get_initial_noise(
        self, batch_size: int, H: int = 512, W: int = 512
    ) -> torch.Tensor:
        return (
            torch.randn(
                batch_size, self.unet.in_channels, H // 8, W // 8, device=self.device
            )
            * self.scheduler.init_noise_sigma
        )

    def predict_noise(self, latents, t, text_embeds):
        return self.unet(latents, t, encoder_hidden_states=text_embeds).sample

    def get_all_step_results(
        self, text_embeddings: torch.Tensor, num_steps: int = 50
    ) -> Tuple[List[torch.Tensor], List[float]]:
        B = text_embeddings.shape[0]
        all_denoised = []
        all_t = []
        latents = self.get_initial_noise(B)
        self.scheduler.set_timesteps(num_steps)
        for t in self.scheduler.timesteps:
            latents = self.denoise_step(latents, t, text_embeddings, 1.0)
            all_denoised.append(latents)
            all_t.append(t - 2)
        return all_denoised, all_t

    def denoise_step(self, latents, t, text_embeds, guidance_scale=7.5):

        # Classifier-free guidance support
        if text_embeds.shape[0] == 2 * latents.shape[0]:
            latent_in = torch.cat([latents, latents])
        else:
            latent_in = latents
        noise_pred = self.unet(latent_in, t, encoder_hidden_states=text_embeds).sample

        if text_embeds.shape[0] == 2 * latents.shape[0]:
            noise_uncond, noise_pos = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_pos - noise_uncond)

        step = self.scheduler.step(noise_pred, t, latents)
        return step.prev_sample

    @torch.no_grad()
    def ddim_invert(self, image, num_inversion_steps=50, prompt="", guidance_scale=7.5):
        emb = self.embed_text(prompt)
        latents = self.encode_image(image)
        self.scheduler.set_timesteps(num_inversion_steps)

        for t in self.scheduler.timesteps:
            latents = self.denoise_step(latents, t, emb, guidance_scale)

        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=30,
        guidance_scale=7.5,
        **kwargs,
    ):
        return super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
