from pathlib import Path

import torch

from script.config import OUTPUT_ROOT


def output_dir() -> Path:
    path = Path(OUTPUT_ROOT)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tensor(tensor: torch.Tensor, name: str) -> Path:
    path = output_dir() / name
    torch.save(tensor.cpu(), path)
    return path


def load_tensor(name: str) -> torch.Tensor:
    return torch.load(output_dir() / name, map_location="cpu")
