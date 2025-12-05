"""Training script for the CNN+LSTM ArtEmis baseline."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.artemis_dataset import ArtEmisCaptionDataset, collate_captions
from src.utils.tokenization import Tokenizer
from src.models.cnn_lstm.cnn_lstm_model import ImageCaptioningCNNLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+LSTM caption model")
    parser.add_argument("--csv", dest="csv_path", type=str, required=True, help="Path to combined captions CSV")
    parser.add_argument("--images-root", type=str, required=True, help="Root folder with resized WikiArt images")
    parser.add_argument("--vocab-path", type=str, default="src/utils/vocab.json", help="Shared vocab JSON from tokenization.py")
    parser.add_argument("--split-path", type=str, default=None, help="Optional path to cached JSON split file")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency for vocab")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--image-feat-dim", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=40, help="Max caption length including BOS/EOS tokens")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to store checkpoints")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (set 0 on Windows)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA")
    parser.add_argument("--max-grad-norm", type=float, default=5.0, help="Gradient clipping norm")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Stop if val loss stalls")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="LR factor for plateau scheduler")
    parser.add_argument("--scheduler-patience", type=int, default=2, help="Plateau scheduler patience")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--log-file", type=str, default=None, help="Optional CSV log output")
    parser.add_argument("--embedding-checkpoint", type=str, default=None, help="Optional .pt file with 'embedding_matrix' to initialize decoder embeddings (TF-IDF / GloVe / FastText)")
    parser.add_argument("--freeze-embedding", action="store_true", help="Freeze embedding layer after loading checkpoint inputs")
    parser.add_argument(
        "--embedding-unfreeze-epoch",
        type=int,
        default=None,
        help="Epoch number (1-indexed) at which to unfreeze the embedding layer; ignored if not frozen",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    # align every rng so experiments are repeatable
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(vocab_path: Path) -> Tokenizer:
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Missing vocab at {vocab_path}. Please run src/utils/tokenization.py to build it first."
        )
    return Tokenizer.load(vocab_path)


def build_or_load_splits(csv_path: Path, split_path: Path, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, list[str]]:
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    df = pd.read_csv(csv_path)
    if "painting" not in df.columns:
        raise ValueError("Expected 'painting' column in CSV for splitting")

    unique_images = df["painting"].dropna().astype(str).unique().tolist()
    if not unique_images:
        raise ValueError("No image identifiers found for split generation")

    rng = random.Random(seed)
    rng.shuffle(unique_images)
    # shuffle once so every split stays fixed on disk

    total = len(unique_images)
    test_count = max(1, int(total * test_ratio)) if test_ratio > 0 else 0
    val_count = max(1, int(total * val_ratio)) if val_ratio > 0 else 0
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Train split would be empty; adjust val/test ratios")

    splits = {
        "train": unique_images[:train_count],
        "val": unique_images[train_count : train_count + val_count],
        "test": unique_images[train_count + val_count : train_count + val_count + test_count],
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as fp:
        json.dump(splits, fp, indent=2)
    return splits


class EarlyStopping:
    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if metric < self.best:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        # stop when validation refuses to improve for patience rounds
        return self.counter >= self.patience


def init_log_file(log_path: Path) -> None:
    if log_path.exists():
        return
    with open(log_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "grad_norm"])


def append_log_row(log_path: Path, row: Iterable[float]) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(row)


def maybe_load_embedding_checkpoint(
    model: ImageCaptioningCNNLSTM,
    checkpoint_path: Path | None,
    freeze: bool,
) -> bool:
    embedding = model.decoder.embedding
    if checkpoint_path is None:
        if freeze:
            embedding.weight.requires_grad_(False)
            return True
        return False

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Embedding checkpoint not found at {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    matrix = payload.get("embedding_matrix")
    if matrix is None:
        raise ValueError("Checkpoint is missing 'embedding_matrix'.")
    if matrix.shape != embedding.weight.data.shape:
        raise ValueError(
            "Embedding checkpoint shape mismatch: "
            f"expected {embedding.weight.data.shape}, got {tuple(matrix.shape)}"
        )
    embedding.weight.data.copy_(matrix.to(embedding.weight.device))
    embedding.weight.requires_grad_(not freeze)
    return freeze


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    max_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device)
        captions_in = batch["captions_in"].to(device)
        captions_out = batch["captions_out"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            logits = model(images, captions_in)
            loss = criterion(logits.view(-1, logits.size(-1)), captions_out.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)  # tame exploding grads early
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_grad_norm += float(grad_norm)
        steps += 1

    mean_loss = total_loss / max(1, steps)
    mean_grad = total_grad_norm / max(1, steps)
    return mean_loss, mean_grad


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images = batch["images"].to(device)
            captions_in = batch["captions_in"].to(device)
            captions_out = batch["captions_out"].to(device)
            with autocast(enabled=use_amp):
                logits = model(images, captions_in)
                loss = criterion(logits.view(-1, logits.size(-1)), captions_out.view(-1))
            total_loss += loss.item()
            steps += 1
    return total_loss / max(1, steps)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
    history: list[dict[str, float]],
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "val_loss": val_loss,
        "history": history,
    }
    ckpt_path = output_dir / f"cnn_lstm_epoch{epoch:03d}_val{val_loss:.3f}.pt"
    torch.save(ckpt, ckpt_path)

    vocab_copy = output_dir / f"vocab_epoch{epoch:03d}.json"
    tokenizer.save(vocab_copy)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    images_root = Path(args.images_root)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    if args.embedding_unfreeze_epoch is not None and args.embedding_unfreeze_epoch < 1:
        raise ValueError("embedding-unfreeze-epoch must be >= 1 when provided")

    tokenizer = load_tokenizer(vocab_path)
    split_path = Path(args.split_path) if args.split_path else csv_path.parent / "splits.json"
    splits = build_or_load_splits(csv_path, split_path, args.val_ratio, args.test_ratio, args.seed)

    train_dataset = ArtEmisCaptionDataset(
        csv_path=csv_path,
        img_root=images_root,
        tokenizer=tokenizer,
        max_len=args.max_len,
        image_filter=set(splits["train"]),
    )
    val_dataset = ArtEmisCaptionDataset(
        csv_path=csv_path,
        img_root=images_root,
        tokenizer=tokenizer,
        max_len=args.max_len,
        image_filter=set(splits["val"]),
    )
    test_ids = set(splits.get("test", []))
    test_dataset = None
    if test_ids:
        test_dataset = ArtEmisCaptionDataset(
            csv_path=csv_path,
            img_root=images_root,
            tokenizer=tokenizer,
            max_len=args.max_len,
            image_filter=test_ids,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_captions,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_captions,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_captions,
            persistent_workers=args.num_workers > 0,
        )

    device = torch.device(args.device)
    model = ImageCaptioningCNNLSTM(
        vocab_size=len(tokenizer.word2idx),
        image_feature_dim=args.image_feat_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    embedding_ckpt = Path(args.embedding_checkpoint) if args.embedding_checkpoint else None
    embedding_frozen = maybe_load_embedding_checkpoint(model, embedding_ckpt, args.freeze_embedding)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = Adam(model.parameters(), lr=args.lr)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)  # amp speeds up larger batches on gpu
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    early_stopper = EarlyStopping(patience=args.early_stop_patience)

    log_path = Path(args.log_file) if args.log_file else output_dir / "training_log.csv"
    init_log_file(log_path)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if (
            embedding_frozen
            and args.embedding_unfreeze_epoch is not None
            and epoch >= args.embedding_unfreeze_epoch
        ):
            model.decoder.embedding.weight.requires_grad_(True)
            embedding_frozen = False
            print(f"[train] Unfroze embedding layer at epoch {epoch}")

        train_loss, grad_norm = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, args.max_grad_norm)
        val_loss = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        current_lr = optimizer.param_groups[0]["lr"]
        append_log_row(log_path, [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.6e}", f"{grad_norm:.4f}"])
        print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(output_dir, model, tokenizer, args, epoch, val_loss, history)

        if early_stopper.step(val_loss):
            print("early stopping triggered")
            break

    if test_loader is not None:
        test_loss = evaluate(model, test_loader, criterion, device, use_amp)
        print(f"test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
