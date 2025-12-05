from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split

from src.datasets.artemis_dataset import ArtemisDataset
from src.models.vit.caption_transformer import CaptionTransformer
from src.models.vit.config_transformer import TransformerHyperParams
from src.utils.tokenization import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ViT-based caption transformer.")
    parser.add_argument("--csv", type=Path, default=Path("data/artemis_sample.csv"))
    parser.add_argument("--images-root", type=Path, default=Path("data/wikiart_sample_128"))
    parser.add_argument("--vocab-path", type=Path, default=Path("src/utils/vocab.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/outputfiles"),
        help="Directory for saving ViT checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters (fallbacks match config defaults)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=32)

    return parser.parse_args()


def collate_batch(batch):
    images, captions_in, captions_out = zip(*batch)
    return {
        "images": torch.stack(images, dim=0),
        "captions_in": torch.stack(captions_in, dim=0),
        "captions_out": torch.stack(captions_out, dim=0),
    }


def create_dataloaders(dataset: ArtemisDataset, val_ratio: float, batch_size: int, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: CaptionTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    max_grad_norm: float,
    log_interval: int,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device, non_blocking=True)
        captions_in = batch["captions_in"].to(device, non_blocking=True)
        captions_out = batch["captions_out"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images, captions_in)
            loss = criterion(logits.view(-1, logits.size(-1)), captions_out.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if step % log_interval == 0:
            logging.info("train step=%d loss=%.4f", step, loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model: CaptionTransformer, loader: DataLoader, criterion: nn.Module, device: torch.device, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        captions_in = batch["captions_in"].to(device, non_blocking=True)
        captions_out = batch["captions_out"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images, captions_in)
            loss = criterion(logits.view(-1, logits.size(-1)), captions_out.view(-1))
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def save_checkpoint(state: Dict, output_dir: Path, epoch: int, val_loss: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"vit_epoch{epoch:03d}_val{val_loss:.3f}.pt"
    torch.save(state, ckpt_path)
    logging.info("Saved checkpoint -> %s", ckpt_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")

    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    tokenizer = Tokenizer.load(args.vocab_path)
    hyper = TransformerHyperParams(
        vocab_size=len(tokenizer.word2idx),
        pad_idx=tokenizer.pad_idx,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        patch_size=args.patch_size,
        max_seq_len=args.max_len,
    )

    dataset = ArtemisDataset(
        csv_path=args.csv,
        img_root=args.images_root,
        tokenizer=tokenizer,
        max_len=hyper.max_seq_len,
    )

    train_loader, val_loader = create_dataloaders(
        dataset,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CaptionTransformer(
        vision_config=hyper.vision_config(),
        decoder_config=hyper.decoder_config(),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.max_grad_norm,
            args.log_interval,
            use_amp,
        )
        val_loss = evaluate(model, val_loader, criterion, device, use_amp)

        logging.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "hyperparams": hyper.__dict__,
                },
                args.output_dir,
                epoch,
                val_loss,
            )


if __name__ == "__main__":
    main()

'''
python -m src.training.train_vit \
  --csv data/artemis_sample.csv \
  --images-root data/wikiart_sample_128 \
  --vocab-path src/utils/vocab.json \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-4
'''