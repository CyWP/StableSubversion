import argparse
import json
import logging
import sys
import yaml

from pathlib import Path
from typing import Dict, Any

from .config import Config
from .training import Trainer


def load_config_from_file(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        return yaml.safe_load(text) or {}
    else:
        return json.loads(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stable-subversion", description="StableSubversion CLI"
    )
    p.add_argument("--config", "-c", type=Path, help="Path to JSON/YAML config file")
    p.add_argument("--train", action="store_true", help="Run training script")
    p.add_argument(
        "--generate",
        "-g",
        type=Path,
        help="Path to pretrained model to generate from config",
    )
    p.add_argument("--info", action="store_true", help="Show config options")
    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config_file:
        config = Config(load_config_from_file(args.config_file))
    else:
        config = Config()

    if args.info:
        print("Configuration:")
        print(json.dumps(config, indent=2))
        return 0

    if args.train:
        Trainer(config).train_lora()


if __name__ == "__main__":
    raise SystemExit(main())
