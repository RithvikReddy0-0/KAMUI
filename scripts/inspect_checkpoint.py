"""Inspect a KAMUI checkpoint: print config, parameter counts, and losses.

Usage:
    python scripts/inspect_checkpoint.py --checkpoint checkpoints/run/step_0000200.pt
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a KAMUI checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"step:       {ckpt['step']}")
    print(f"train_loss: {ckpt.get('train_loss')}")
    print(f"val_loss:   {ckpt.get('val_loss')}")
    if ckpt.get("config"):
        print("config:")
        for key, value in ckpt["config"].items():
            print(f"  {key}: {value}")
    n_params = sum(v.numel() for v in ckpt["model_state"].values())
    print(f"parameters (incl. buffers): {n_params:,}")
    print("modules:")
    for name, tensor in ckpt["model_state"].items():
        print(f"  {name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
