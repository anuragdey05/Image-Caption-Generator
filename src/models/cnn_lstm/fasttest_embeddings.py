"""Build FastText embeddings aligned with the shared vocab for CNN+LSTM."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.utils.tokenization import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN

SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


@dataclass
class FastTextStats:
    vocab_size: int
    embedding_dim: int
    matched_tokens: int
    token_coverage: float


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    word2idx = {token: int(idx) for token, idx in payload["word2idx"].items()}
    if not word2idx:
        raise ValueError(f"Vocabulary at {vocab_path} is empty.")
    return word2idx


def _default_token_variants(token: str) -> list[str]:
    """Simple normalization variants to better match your vocab."""
    variants = [token, token.lower()]
    if "’" in token:
        variants.append(token.replace("’", "'"))
    if token.startswith('"') and token.endswith('"'):
        variants.append(token.strip('"'))
    return variants


def load_fasttext_subset(
    fasttext_path: Path,
    vocab: Dict[str, int],
    expected_dim: int | None,
) -> tuple[Dict[str, np.ndarray], int]:
    """
    Load a FastText *binary* model and fetch vectors for all vocab tokens.

    This uses FastText's subword mechanism: get_word_vector() returns a vector
    for ANY token, so there are effectively no OOVs here.
    """
    import fasttext  # requires `pip install fasttext`

    model = fasttext.load_model(str(fasttext_path))
    model_dim = model.get_dimension()
    embedding_dim = expected_dim if expected_dim is not None else model_dim
    if embedding_dim != model_dim:
        raise ValueError(
            f"Requested embedding_dim={embedding_dim}, "
            f"but FastText model dimension is {model_dim}. "
            "Use a projection layer or reduce offline if you need a different dim."
        )

    vectors: Dict[str, np.ndarray] = {}
    for token in vocab.keys():
        # Try basic normalization variants to align with your vocab
        for candidate in _default_token_variants(token):
            vec = model.get_word_vector(candidate)
            # FastText always returns a vector, but if your vocab uses unusual
            # tokens you can add extra checks here; we'll just accept it.
            if vec is not None:
                vectors[token] = np.asarray(vec, dtype=np.float32)
                break
        else:
            # Fallback in the extremely rare case we didn't break above
            vec = model.get_word_vector(token)
            vectors[token] = np.asarray(vec, dtype=np.float32)

    return vectors, embedding_dim


def build_embedding_matrix(
    vocab: Dict[str, int],
    fasttext_vectors: Dict[str, np.ndarray],
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, FastTextStats]:
    # Random init is now mostly redundant (we overwrite all tokens),
    # but we keep it for consistency / safety.
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(len(vocab), embedding_dim)).astype(np.float32)

    hits = 0
    for token, idx in vocab.items():
        vector = fasttext_vectors.get(token)
        if vector is None:
            continue
        matrix[idx] = vector
        hits += 1

    # In practice, hits should == len(vocab) and coverage == 1.0
    hit_indices = [
        idx for token, idx in vocab.items()
        if token not in SPECIAL_TOKENS and token in fasttext_vectors
    ]
    mean_vector = (
        matrix[hit_indices].mean(axis=0)
        if hit_indices
        else np.zeros(embedding_dim, dtype=np.float32)
    )
    for token in SPECIAL_TOKENS:
        idx = vocab.get(token)
        if idx is None:
            continue
        if token == PAD_TOKEN:
            matrix[idx] = 0.0
        else:
            matrix[idx] = mean_vector

    coverage = hits / float(len(vocab))
    stats = FastTextStats(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        matched_tokens=hits,
        token_coverage=coverage,
    )
    return matrix, stats


def save_embeddings(matrix: np.ndarray, stats: FastTextStats, output_path: Path) -> None:
    tensor = torch.from_numpy(matrix)
    torch.save(
        {
            "embedding_matrix": tensor,
            "stats": {
                "vocab_size": stats.vocab_size,
                "embedding_dim": stats.embedding_dim,
                "matched_tokens": stats.matched_tokens,
                "token_coverage": stats.token_coverage,
            },
        },
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align FastText vectors with the shared CNN+LSTM vocabulary."
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("src/utils/vocab.json"),
        help="Shared vocabulary JSON produced by tokenization.py.",
    )
    parser.add_argument(
        "--fasttext-path",
        type=Path,
        required=True,
        help="Path to the FastText *binary* model (e.g., cc.en.300.bin).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help=(
            "Expected embedding dimension. Leave blank to infer from the FastText model. "
            "If set, must match model.get_dimension()."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/fasttext_embeddings.pt"),
        help="Destination .pt file storing the aligned embedding matrix.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initializing OOV token vectors (now mostly unused).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = load_vocab(args.vocab_path)
    fasttext_vectors, inferred_dim = load_fasttext_subset(args.fasttext_path, vocab, args.embedding_dim)
    matrix, stats = build_embedding_matrix(vocab, fasttext_vectors, inferred_dim, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(matrix, stats, args.output)

    print(
        f"[FastText] vocab={stats.vocab_size:,} | dim={stats.embedding_dim} | "
        f"hits={stats.matched_tokens:,} | coverage={stats.token_coverage:.3f}"
    )
    print(f"[FastText] Embedding matrix saved to {args.output}")


if __name__ == "__main__":
    main()

# example usage:
# python -m src.models.cnn_lstm.fasttest_embeddings \
#   --vocab-path src/utils/vocab.json \
#   --fasttext-path data/wiki-news-300d-1M-subword.bin \
#   --output src/outputfiles/fasttext_embeddings.pt
