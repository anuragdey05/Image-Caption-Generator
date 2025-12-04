from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms


class ImageProcessor:
    """Reusable image loader for ViT and CNN+LSTM models."""

    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        mean: Tuple[float, float, float] | None = None,
        std: Tuple[float, float, float] | None = None,
    ) -> None:
        """Resize + tensor conversion, optional normalization if mean/std provided."""
        steps = [transforms.Resize(size, interpolation=Image.BICUBIC), transforms.ToTensor()]
        if mean is not None and std is not None:
            steps.append(transforms.Normalize(mean, std))
        self.transform = transforms.Compose(steps)

    def load(self, path: Path | str) -> torch.Tensor:
        """Load an RGB image and return a (3, H, W) tensor."""
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))


# Default singleton hooked into datasets
DEFAULT_IMAGE_PROCESSOR = ImageProcessor()


def load_image(path: Path | str) -> torch.Tensor:
    """Functional wrapper used across the codebase."""
    return DEFAULT_IMAGE_PROCESSOR.load(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick check for ImageProcessor")
    parser.add_argument("image", type=str, help="Path to an image file")
    parser.add_argument("--mean", nargs=3, type=float, default=None, help="Optional mean for normalization")
    parser.add_argument("--std", nargs=3, type=float, default=None, help="Optional std for normalization")
    args = parser.parse_args()

    processor = ImageProcessor(mean=tuple(args.mean) if args.mean else None,
                               std=tuple(args.std) if args.std else None)
    tensor = processor.load(args.image)
    print("Loaded image tensor shape:", tensor.shape)
    print("Range: [", tensor.min().item(), ",", tensor.max().item(), "]")