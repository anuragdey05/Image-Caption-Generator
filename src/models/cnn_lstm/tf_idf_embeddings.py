"""Build TF-IDF embeddings aligned with the shared vocab for CNN+LSTM."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.tokenization import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    Tokenizer,
)

SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


@dataclass
class TFIDFStats:
    vocab_size: int
    embedding_dim: int
    token_coverage: float
    doc_count: int


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    word2idx = {token: int(idx) for token, idx in payload["word2idx"].items()}
    if not word2idx:
        raise ValueError(f"Vocabulary at {vocab_path} is empty.")
    return word2idx


def load_captions(csv_path: Path, preferred_column: str, max_examples: int | None) -> List[str]:
    df = pd.read_csv(csv_path)
    candidate_columns = [preferred_column, "utterance", "caption"]
    column = next((col for col in candidate_columns if col in df.columns), None)
    if column is None:
        raise ValueError(
            f"None of the expected caption columns {candidate_columns} exist in {csv_path}"
        )
    texts = df[column].dropna().astype(str).tolist()
    if max_examples is not None:
        texts = texts[:max_examples]
    # Reuse the shared cleaner so TF-IDF sees the same normalization as the vocab builder
    return [Tokenizer.clean_text(text) for text in texts]


def build_token_embeddings(
    texts: Sequence[str],
    vocab: Dict[str, int],
    embedding_dim: int,
    min_df: int,
    max_df: float,
    seed: int,
) -> tuple[np.ndarray, TFIDFStats]:
    # Fix the vectorizer vocabulary so rows line up exactly with the CNN+LSTM indices
    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
    )
    doc_term = vectorizer.fit_transform(texts)
    if doc_term.shape[0] == 0:
        raise RuntimeError("TF-IDF fit produced an empty matrix. Check the input captions.")

    token_doc = doc_term.transpose().tocsr()
    vocab_size, doc_count = token_doc.shape
    non_zero = token_doc.getnnz(axis=1)
    coverage = float((non_zero > 0).sum()) / float(vocab_size)

    # Optional dimensionality reduction keeps the matrix manageable for the embedding layer
    target_dim = min(max(1, embedding_dim), doc_count) if embedding_dim > 0 else doc_count
    if target_dim < doc_count:
        reducer = TruncatedSVD(n_components=target_dim, random_state=seed)
        embeddings = reducer.fit_transform(token_doc)
    else:
        embeddings = token_doc.toarray()

    embeddings = embeddings.astype(np.float32)

    # Ensure special tokens behave as expected when copied into nn.Embedding weights
    regular_indices = [idx for token, idx in vocab.items() if token not in SPECIAL_TOKENS]
    mean_vector = (
        embeddings[regular_indices].mean(axis=0)
        if regular_indices
        else np.zeros(embeddings.shape[1], dtype=np.float32)
    )
    for token in SPECIAL_TOKENS:
        token_idx = vocab.get(token)
        if token_idx is None or token_idx >= embeddings.shape[0]:
            continue
        if token == PAD_TOKEN:
            embeddings[token_idx] = 0.0
        else:
            embeddings[token_idx] = mean_vector

    stats = TFIDFStats(
        vocab_size=vocab_size,
        embedding_dim=embeddings.shape[1],
        token_coverage=coverage,
        doc_count=doc_count,
    )
    return embeddings, stats


def save_embeddings(matrix: np.ndarray, stats: TFIDFStats, output_path: Path) -> None:
    tensor = torch.from_numpy(matrix)
    torch.save(
        {
            "embedding_matrix": tensor,
            "stats": {
                "vocab_size": stats.vocab_size,
                "embedding_dim": stats.embedding_dim,
                "token_coverage": stats.token_coverage,
                "doc_count": stats.doc_count,
            },
        },
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute TF-IDF embeddings that plug directly into the CNN+LSTM decoder."
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("src/utils/vocab.json"),
        help="Shared vocabulary JSON produced by tokenization.py",
    )
    parser.add_argument(
        "--captions-csv",
        type=Path,
        default=Path("data/artemis_sample.csv"),
        help="Caption CSV used to fit TF-IDF statistics (e.g., ArtEmis subset).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="utterance",
        help="Preferred column name containing caption text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tfidf_embeddings.pt"),
        help="Destination .pt file. Load and copy into nn.Embedding before training.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Target embedding dimension after optional SVD compression.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency to keep a token active in TF-IDF stats.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Maximum document frequency (fraction) to retain a token.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap the number of captions for quicker experimentation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for truncated SVD (PCA-style reduction).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = load_vocab(args.vocab_path)
    captions = load_captions(args.captions_csv, args.text_column, args.max_examples)
    if not captions:
        raise RuntimeError(f"No valid captions found in {args.captions_csv}.")

    matrix, stats = build_token_embeddings(
        texts=captions,
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        min_df=args.min_df,
        max_df=args.max_df,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(matrix, stats, args.output)

    print(
        f"[TF-IDF] vocab={stats.vocab_size:,} | dim={stats.embedding_dim} | "
        f"coverage={stats.token_coverage:.3f} | docs={stats.doc_count:,}"
    )
    print(f"[TF-IDF] Embedding matrix saved to {args.output}")


if __name__ == "__main__":
    main()

#example - added by soubhagya
# python -m src.models.cnn_lstm.tf_idf_embeddings \
#   --vocab-path src/utils/vocab.json \
#   --captions-csv data/artemis_sample.csv \
#   --output models/tfidf_embeddings.pt \
#   --embedding-dim 256