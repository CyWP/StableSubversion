import torch

from .utils import EasyDict


class Config(EasyDict):

    _defaults = {
        "run_dir": ".",
        "run_name": "subvert_test_run",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "target_prompt": None,
        "epochs": 100,
        "batch_size": 64,
        "minibatch_size": 4,
        "train_size": 1000,
        "test_size": 50,
        "lora_rank": 8,
        "lr": 1e-4,
        "min_steps": 25,
        "max_steps": 50,
        "mixed_precision": "fp16",
        "clip_name": "openai/clip-vit-large-patch14",
        "distillation_weight": 0.5,
        "adversarial_weight": 0.5,
        "test_interval": 1,
        "test_inference_steps": 30,
        "test_cfg": 1.0,
        "seed": 42,
        "sched_cycle": 20,
    }

    def __init__(self, **kwargs):
        super().__init__(Config._defaults)
        self.update(kwargs)
        self.generator = torch.Generator().manual_seed(self.seed)
