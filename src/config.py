import torch


class Config(dict):

    _defaults = {
        "run_dir": ".",
        "run_name": "subvert_test_run",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "target_prompt": None,
        "epochs": 100,
        "batch_size": 16,
        "train_size": 1000,
        "test_size": 50,
        "lora_rank": 8,
        "lora_alpha": 16,
        "train_batch_size": 4,
        "lr": 1e-4,
        "min_steps": 25,
        "max_steps": 100,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 1,
        "clip_name": "openai/clip-vit-large-patch14",
        "distillation_weight": 0.4,
        "target_weight": 0.4,
        "adversarial_weight": 0.2,
        "test_interval": 5,
        "test_inference_steps": 30,
        "test_cfg": 1.0,
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
