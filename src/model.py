import torch

from contextlib import contextmanager
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from pathlib import Path
from typing import Generator, List, Tuple, Optional


class StableSubversionPipeline(StableDiffusionPipeline):

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        revision: str | None = None,
        variant: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
        use_safetensors: bool = True,
        vae: str | None = None,
        device: str = "cuda",
        **kwargs,
    ):

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
            use_safetensors=use_safetensors,
            **({"vae": vae} if vae else {}),
            **kwargs,
        )
        # Replace default scheduler with DDIM
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        self.__dict__.update(pipe.__dict__)
        self._lora_layers = {}
        self._current_lora = None
        self._base_attn_procs = {k: v for k, v in self.unet.attn_processors.items()}

        self.to(device)

    def create_lora(self, name: str, rank: int = 8):
        if name in self._lora_layers:
            raise ValueError(f"LoRA '{name}' already exists.")

        unet_lora = AttnProcsLayers.from_unet(self.unet, rank=rank)

        self._lora_layers[name] = unet_lora
        self._current_lora = name

        self.unet.set_attn_processor(unet_lora)

        return [p for p in unet_lora.parameters() if p.requires_grad]

    def save_lora(self, path: Path, name: Optional[str] = None):
        name = self._current_lora if name is None else name
        if name not in self._lora_layers:
            raise ValueError(f"LoRA '{name}' doesn't exist.")
        torch.save(self._lora_layers[name].state_dict(), path)

    def load_lora(self, name: str, path: str):
        state = torch.load(path, map_location="cpu")

        # Detect rank
        sample_key = next(k for k in state.keys() if "lora_down.weight" in k)
        detected_rank = state[sample_key].shape[0]

        # Rebuild LoRA layers
        unet_loras = AttnProcsLayers.from_unet(self.unet, rank=detected_rank)

        # Load weights
        missing, unexpected = unet_loras.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")

        # Move to device
        for p in unet_loras.parameters():
            p.data = p.data.to(self.device)
            if p._grad is not None:
                p._grad = p._grad.to(self.device)

        self._lora_layers[name] = unet_loras
        self._current_lora = name

        self.unet.set_attn_processor(unet_loras)

        return [p for p in unet_loras.parameters() if p.requires_grad]

    def enable_lora(self, name: str):
        if name not in self._lora_layers:
            raise ValueError(f"LoRA '{name}' not found")
        self.unet.set_attn_processor(self._lora_layers[name])
        self._current_lora = name

    def disable_lora(self):
        self.unet.set_attn_processor(self._base_attn_procs)
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
            all_t.append(t)
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
