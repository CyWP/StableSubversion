import torch
import torch.nn.functional as F
import inspect

from contextlib import contextmanager
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
)
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ScalableAttnProc(LoRAAttnProcessor2_0):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = 1.0

    def set_scale(self, scale: float):
        self.scale = scale

    def get_params(self) -> List[torch.Tensor]:
        params = []
        for layer in [self.to_q_lora, self.to_k_lora, self.to_v_lora, self.to_out_lora]:
            params.append(layer.up.weight)
            params.append(layer.down.weight)
        return params

    def __call__(self, attn, hidden_states, *args, **kwargs):

        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device) * self.scale
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device) * self.scale
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device) * self.scale
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        attn._modules.pop("processor")
        attn.processor = AttnProcessor2_0()
        return attn.processor(attn, hidden_states, *args, **kwargs)


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
        self._base_procs = dict(self.unet.attn_processors)
        self._active_procs = None
        self._lora_layers = {}
        self._current_lora = None

    def print_pipeline_params(self):
        print("inspect")
        for module_name in ["unet", "vae", "text_encoder"]:
            module = getattr(self, module_name, None)
            if module is not None:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        print(
                            f"{module_name}.{name}: requires_grad={param.requires_grad}"
                        )

    def ensure_name(self, name: str) -> str:
        if name is None:
            if self._current_lora is None:
                raise ValueError(f"No LoRA has been initialized yet.")
            return self._current_lora
        if name not in self._lora_layers:
            raise ValueError(f"LoRA '{name}' not found")
        return name

    def get_lora_procs(self) -> Dict[str, ScalableAttnProc]:
        procs = {}
        for k, proc in self.unet.attn_processors.items():
            if isinstance(proc, ScalableAttnProc):
                procs[k] = proc
        return procs

    def get_lora_params(self) -> List[torch.Tensor]:
        params = []
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, ScalableAttnProc):
                params += proc.get_params()
        return params

    def create_lora(self, name: str, rank: int = 8) -> List[torch.Tensor]:
        if name in self._lora_layers:
            raise ValueError(f"LoRA '{name}' already exists.")
        unet = self.unet
        lora_procs = {}
        for n, base in self._base_procs.items():
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
            lora_procs[n] = ScalableAttnProc(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            )
            print(n, ": ", lora_procs[n])
        self._current_lora = name
        self._lora_layers[name] = lora_procs
        self.enable_lora(name)
        return self.get_lora_params()

    def inspect_lora_weights(self):
        if self._current_lora is None:
            print("No LoRA enabled.")
            return
        for name, proc in self._lora_layers[self._current_lora].items():
            print(name)
            if isinstance(proc, LoRAAttnProcessor2_0):
                up = proc.lora_up.weight
                down = proc.lora_down.weight
                print(
                    f"{name}: lora_up mean={up.mean().item():.6f}, std={up.std().item():.6f}"
                )
                print(
                    f"{name}: lora_down mean={down.mean().item():.6f}, std={down.std().item():.6f}"
                )

    def load_lora(self, path: Path, name: Optional[str] = None):
        loaded_state = torch.load(path, map_location=self.device)
        name = name or path.stem
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
        return [p for p in unet_lora_attn_procs.values()]

    def save_lora(self, path: Path, name: Optional[str] = None):
        name = self.ensure_name(name)
        # gather all processor state dicts
        if name == self._current_lora:
            self.store_lora()
        state = {k: v.state_dict() for k, v in self._lora_layers[name].items()}
        torch.save(state, path)

    def store_lora(self, name: Optional[str] = None):
        name = self.ensure_name(name)
        self._lora_layers[name] = self.get_lora_procs()

    def enable_lora(self, name: str):
        if name not in self._lora_layers:
            raise ValueError(f"LoRA '{name}' not found")
        to_load = {}
        for k, v in self._lora_layers[name].items():
            to_load[k] = v
        self.unet.set_attn_processor(to_load)
        self._current_lora = name

    def disable_lora(self):
        if self._current_lora is None:
            return
        self.store_lora()
        procs = {}
        for k, v in self._base_procs.items():
            procs[k] = v
        self.unet.set_attn_processor(procs)
        self._current_lora = None

    def set_lora_scale(self, scale: float):
        # Only affects currently enabled LoRA
        if self._current_lora is None:
            return
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, ScalableAttnProc):
                proc.set_scale(scale)

    @contextmanager
    def lora_disabled(self):
        self.set_lora_scale(0)
        try:
            yield self
        finally:
            self.set_lora_scale(1)

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
