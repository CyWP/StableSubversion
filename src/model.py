import torch
import torch.nn.functional as F
import inspect

from contextlib import contextmanager
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import (
    Attention,
    LoRAAttnProcessor2_0,
)
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ToggledAttnProc(LoRAAttnProcessor2_0):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def get_params(self) -> List[torch.Tensor]:
        params = []
        for layer in [self.to_q_lora, self.to_k_lora, self.to_v_lora, self.to_out_lora]:
            params.append(layer.up.weight)
            params.append(layer.down.weight)
        return params

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
    ):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = (
            attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
            if self.enabled
            else attn.to_q(hidden_states)
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = (
            attn.to_k(encoder_hidden_states)
            + scale * self.to_k_lora(encoder_hidden_states)
            if self.enabled
            else attn.to_k(encoder_hidden_states)
        )
        value = (
            attn.to_v(encoder_hidden_states)
            + scale * self.to_v_lora(encoder_hidden_states)
            if self.enabled
            else attn.to_v(encoder_hidden_states)
        )

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


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
        requires_safety_checker: bool = False,
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

    def print_lora_params(self):
        print("inspect")
        for module_name in ["unet", "vae", "text_encoder"]:
            module = getattr(self, module_name, None)
            if module is not None:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        print(
                            f"{module_name}.{name}: requires_grad={param.requires_grad}"
                        )

    def get_lora_procs(self) -> Dict[str, ToggledAttnProc]:
        procs = {}
        for k, proc in self.unet.attn_processors.items():
            if isinstance(proc, ToggledAttnProc):
                procs[k] = proc
        return procs

    def get_lora_params(self) -> List[torch.Tensor]:
        params = []
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, ToggledAttnProc):
                params += proc.get_params()
        return params

    def create_lora(self, rank: int = 8):
        unet = self.unet
        lora_procs = {}
        device = self.unet.device
        for n, base in self.unet.attn_processors.items():
            cross_attention_dim = (
                None
                if n.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if n.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif n.startswith("up_blocks"):
                block_id = int(n[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif n.startswith("down_blocks"):
                block_id = int(n[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                raise NotImplementedError(
                    "name must start with up_blocks, mid_blocks, or down_blocks"
                )
            lora_procs[n] = ToggledAttnProc(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            ).to(device)
        self.unet.set_attn_processor(lora_procs)
        self.enable_lora()

    def inspect_lora_weights(self):
        for param in self.get_lora_params():
            print(
                f"param: {param.shape}, min: {param.min()}, max: {param.max()}, mean: {param.mean()}"
            )

    def load_lora(self, path: Path):
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
            proc = ToggledAttnProc(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=st["lora_up.weight"].shape[1],
            )
            proc.load_state_dict(st)
            unet_lora_attn_procs[attn_name] = proc
        self.unet.set_attn_processor(unet_lora_attn_procs)

    def save_lora(self, path: Path):
        state = {k: v.state_dict() for k, v in self.get_lora_procs().items()}
        torch.save(state, path)

    def enable_lora(self):
        for proc in self.get_lora_procs().values():
            proc.enable()

    def disable_lora(self):
        for proc in self.get_lora_procs().values():
            proc.disable()

    @contextmanager
    def lora_disabled(self):
        self.disable_lora()
        try:
            yield self
        finally:
            self.enable_lora()

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

    @torch.no_grad()
    def generate_from_latent(
        self,
        prompt,
        negative_prompt=None,
        latents=None,
        num_inference_steps=30,
        guidance_scale=7.5,
        **kwargs,
    ):
        text_embeds = self.embed_text(prompt, negative_prompt)
        B = text_embeds.shape[0] // 2 if negative_prompt else text_embeds.shape[0]
        if latents is None:
            latents = self.get_initial_noise(B)  # shape: [B, C, H/8, W/8]
        else:
            latents = latents.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            latents = self.denoise_step(latents, t, text_embeds, guidance_scale)
        images = self.decode_latents(latents)
        return images
