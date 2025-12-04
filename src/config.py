import torch
import json

from .utils import EasyDict


class Config(EasyDict):

    _defaults = {
        "run_dir": ".",
        "run_name": "subvert_test_run",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "target_prompt": None,
        "use_negative_prompt": False,
        "epochs": 100,
        "batch_size": 64,
        "minibatch_size": 4,
        "train_size": 1000,
        "test_size": 50,
        "lora_rank": 8,
        "lr": 1e-4,
        "sched_step_size": 1,
        "sched_gamma": 0.9,
        "ema_decay": 0.9,
        "min_steps": 25,
        "max_steps": 50,
        "mixed_precision": "fp16",
        "clip_name": "openai/clip-vit-large-patch14",
        "distillation_weight": 0.5,
        "adversarial_weight": 0.5,
        "test_interval": 1,
        "test_inference_steps": 30,
        "test_cfgs": [1.0, 4.0, 7.0],
        "seed": 42,
        "sched_cycle": 20,
        "hf_auth_token": 0,
    }

    def __init__(self, **kwargs):
        super().__init__(Config._defaults)
        self.update(kwargs)
        self.generator = torch.Generator().manual_seed(self.seed)

    def json(self) -> str:
        j_dict = {k: v for k, v in self.items() if k != "generator"}
        return json.dumps(j_dict)
