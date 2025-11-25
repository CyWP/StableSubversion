import torch

from contextlib import contextmanager
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from typing import Generator


class StableSubversionPipeline(StableDiffusionPipeline):
    """
    A Stable Diffusion pipeline that supports:
    - Creating & training LoRA modules
    - Differentiable UNet calls for training
    - Forward diffusion (q(x_t | x_0))
    - Reverse diffusion step (p(x_{t-1}|x_t))
    - DDIM inversion
    - Inference with or without LoRA
    """

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

        self.to(device)

    # ------------------------------------------------------------
    #  LoRA MANAGEMENT — training-ready
    # ------------------------------------------------------------

    def create_lora(self, name: str, rank: int = 8):
        """
        Creates trainable LoRA layers for UNet + text encoder.
        Returns a list of parameters for optimizer use.
        """

        if name in self._lora_layers:
            raise ValueError(f"LoRA '{name}' already exists.")

        unet_lora = AttnProcsLayers.from_unet(self.unet, rank=rank)
        text_lora = AttnProcsLayers.from_text_encoder(self.text_encoder, rank=rank)

        combined = AttnProcsLayers({**unet_lora.layers, **text_lora.layers})
        self._lora_layers[name] = combined
        self._current_lora = name

        # Attach to model
        self.unet.set_attn_processor(combined)

        # return parameters (trainable)
        return [p for p in combined.parameters() if p.requires_grad]

    def load_lora(self, path: str, name: str = "default"):
        self.load_lora_weights(path)
        self._lora_layers[name] = self.unet.attn_processors
        self._current_lora = name

    def enable_lora(self, name: str):
        self.unet.set_attn_processor(self._lora_layers[name])
        self._current_lora = name

    def disable_lora(self):
        base = AttnProcsLayers.from_unet(self.unet, rank=None)
        self.unet.set_attn_processor(base)
        self._current_lora = None

    @contextmanager
    def lora_disabled(self) -> Generator["StableSubversionPipeline"]:
        tmp = self._current_lora
        self.disable_lora()
        try:
            yield self
        except:
            self.enable_lora(tmp)

    def embed_text(self, prompt: str, negative_prompt: str | None = None):
        """
        Return text embeddings for training or inference.
        """

        tok = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        pos = self.text_encoder(tok.input_ids)[0]

        if negative_prompt is None:
            return pos

        neg_tok = self.tokenizer(
            negative_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        neg = self.text_encoder(neg_tok.input_ids)[0]
        return torch.cat([neg, pos], dim=0)

    def encode_image(self, image):
        """
        Encode image into latent x0. Used for training or inversion.
        """
        return self.vae.encode(image).latent_dist.sample() * 0.18215

    def decode_latents(self, latents):
        """
        Decode latents back to image.
        """
        latents = latents / 0.18215
        return self.vae.decode(latents).sample

    def add_noise(self, x0, t, noise):
        """
        Forward diffusion process (training).
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(x0.device)
        a = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sigma = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return a * x0 + sigma * noise

    def predict_noise(self, latents, t, text_embeds):
        """
        One UNet forward pass. Use this inside your training loss.
        """
        return self.unet(latents, t, encoder_hidden_states=text_embeds).sample

    def denoise_step(self, latents, t, text_embeds, guidance_scale=7.5):
        """
        One reverse diffusion step.
        Used for:
            - DDIM reverse sampling
            - Training custom loss terms that need p(x_{t-1}|x_t)
            - Timestep skipping research
        """

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

    # ------------------------------------------------------------
    #  INFERENCE (unchanged)
    # ------------------------------------------------------------

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
