import torch


class Config(dict):

    _defaults = {
        "model_name": "runwayml/stable-diffusion-v1-5",
        "target_prompt": "A golden retriever puppy",
        "epochs": 100,
        "batch_size": 16,
        "train_size": 10000,
        "val_size": 1000,
        "test_size": 1000,
        "lora_rank": 8,
        "lora_alpha": 16,
        "train_batch_size": 4,
        "lr": 1e-4,
        "timestep_min": 10,
        "timestep_max": 990,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 1,
        "clip_name": "openai/clip-vit-large-patch14",
    }

    def __init__(self, **kwargs):
        super().__init__(Config._defaults)
        self.update(kwargs)
        self.generator = torch.Generator().manual_seed(self.seed)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"'Config' object has no attribute '{name}'") from e

    def __setattr__(self, name, value):
        self[name] = self._convert_value(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"'Config' object has no attribute '{name}'") from e
