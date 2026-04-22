"""Device / dtype helpers. Fixed at float32 everywhere (match Fortran oracle)."""
from __future__ import annotations
import os
import torch

DTYPE = torch.float32


def pick_device(prefer: str | None = None) -> torch.device:
    """Pick a CUDA device (`cuda:<i>`) or fall back to CPU.

    Priority: explicit argument → ``PWMVA_TORCH_DEVICE`` env → first CUDA → cpu.
    """
    if prefer is None:
        prefer = os.environ.get("PWMVA_TORCH_DEVICE", None)
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
