from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.image_utils import load_image
from src.utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)


class ArtemisDataset(Dataset):
    """
    WikiArt-ArtEmis pairs producing:
    - image_tensor: (3, 128, 128) for ViT encoder
    - caption_in: decoder inputs (BOS … token_{T-1})
    - caption_out: training targets (token_1 … EOS)
    
    Skips samples with missing images and logs warnings.
    """

    def __init__(
        self,
        csv_path: Path | str,
        img_root: Path | str,
        tokenizer: Tokenizer,
        max_len: int,
        image_loader: Callable[[Path], torch.Tensor] = load_image,
    ) -> None:
        df = pd.read_csv(csv_path)
        if "utterance" not in df.columns:
            raise ValueError("CSV must contain an 'utterance' column.")
        self.img_root = Path(img_root)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_loader = image_loader
        
        # Filter out rows with missing images
        self.valid_indices = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            img_path = self._resolve_image_path(str(row["art_style"]), str(row["painting"]))
            if img_path.exists():
                self.valid_indices.append(idx)
            else:
                logger.warning(f"Missing image: {img_path}, skipping sample {idx}")
        
        self.df = df.iloc[self.valid_indices].reset_index(drop=True)
        logger.info(f"Loaded {len(self.df)}/{len(df)} samples with valid images")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, art_style: str, painting: str) -> Path:
        """Append .jpg if the painting field lacks an extension."""
        painting_name = painting if painting.endswith(('.jpg', '.jpeg', '.png')) else f"{painting}.jpg"
        return self.img_root / art_style / painting_name

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(str(row["art_style"]), str(row["painting"]))
        image_tensor = self.image_loader(img_path)

        caption = str(row["utterance"])
        caption_ids = self.tokenizer.encode_caption(caption, self.max_len)
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        caption_in = caption_tensor[:-1]
        caption_out = caption_tensor[1:]

        return image_tensor, caption_in, caption_out


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "artemis_sample.csv"
    img_root = project_root / "data" / "wikiart_sample_128"
    vocab_path = project_root / "src" / "utils" / "vocab.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sample CSV at {csv_path}")
    if not img_root.exists():
        raise FileNotFoundError(f"Missing image root at {img_root}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab at {vocab_path}. Run tokenization.py first.")

    tokenizer = Tokenizer.load(vocab_path)
    dataset = ArtemisDataset(
        csv_path=csv_path,
        img_root=img_root,
        tokenizer=tokenizer,
        max_len=32,
    )
    print(f"Dataset size: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    images, captions_in, captions_out = next(iter(loader))
    print("Batch image tensor shape:", images.shape)
    print("Batch caption_in shape:", captions_in.shape)
    print("Batch caption_out shape:", captions_out.shape)