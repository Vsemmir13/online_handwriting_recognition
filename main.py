import argparse
import yaml
import torch
import os
import sys
from MODEL.model.worker import launch_training, launch_inference, launch_example
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="PTOHWR App Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer", "example"],
        default="train",
        help="Run mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cuda", "cpu"],
        default="mps",
        help="Device for model",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth/.pt)")
    parser.add_argument("--split", type=str, default="testset_f.txt", help="Split file for inference (uses IAM-OnDB splits)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if args.mode == "train":
        print("Starting training...")
        launch_training(config)
    elif args.mode == "infer":
        assert args.checkpoint, "--checkpoint is required for infer mode"
        print("Running inference...")
        launch_inference(
            config=config,
            checkpoint_path=args.checkpoint,
        )
    elif args.mode == "example":
        assert args.checkpoint, "--checkpoint is required for example mode"
        print("Running single-sample example...")
        launch_example(
            config=config,
            checkpoint_path=args.checkpoint
        )


if __name__ == "__main__":
    main()
