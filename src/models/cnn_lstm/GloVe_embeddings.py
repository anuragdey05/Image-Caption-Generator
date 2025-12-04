"""Load pre-trained GloVe vectors and align them with the shared vocab."""

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
class GloveStats:
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


def load_glove_subset(
	glove_path: Path, vocab: Dict[str, int], expected_dim: int | None
) -> tuple[Dict[str, np.ndarray], int]:
	vectors: Dict[str, np.ndarray] = {}
	embedding_dim = expected_dim
	with glove_path.open("r", encoding="utf-8") as handle:
		for line_num, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			token, coeffs = parts[0], parts[1:]
			if embedding_dim is None:
				embedding_dim = len(coeffs)
			if embedding_dim != len(coeffs):
				# Some GloVe dumps start with a header row (count dim). Skip mismatches.
				continue
			if token not in vocab:
				continue
			vector = np.asarray(coeffs, dtype=np.float32)
			vectors[token] = vector
			if len(vectors) == len(vocab):
				break
	if embedding_dim is None:
		raise ValueError(f"Could not infer embedding dimension from {glove_path}.")
	return vectors, embedding_dim


def build_embedding_matrix(
	vocab: Dict[str, int],
	glove_vectors: Dict[str, np.ndarray],
	embedding_dim: int,
	seed: int,
) -> tuple[np.ndarray, GloveStats]:
	rng = np.random.default_rng(seed)
	matrix = rng.normal(loc=0.0, scale=0.05, size=(len(vocab), embedding_dim)).astype(np.float32)

	hits = 0
	for token, idx in vocab.items():
		vector = glove_vectors.get(token)
		if vector is None:
			continue
		matrix[idx] = vector
		hits += 1

	# Special tokens: keep <pad> zeroed, reuse average vector for others to stabilize decoding
	hit_indices = [vocab[tok] for tok in glove_vectors.keys() if tok not in SPECIAL_TOKENS and tok in vocab]
	mean_vector = (
		matrix[hit_indices].mean(axis=0) if hit_indices else np.zeros(embedding_dim, dtype=np.float32)
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
	stats = GloveStats(
		vocab_size=len(vocab),
		embedding_dim=embedding_dim,
		matched_tokens=hits,
		token_coverage=coverage,
	)
	return matrix, stats


def save_embeddings(matrix: np.ndarray, stats: GloveStats, output_path: Path) -> None:
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
		description="Align pre-trained GloVe vectors with the shared CNN+LSTM vocabulary."
	)
	parser.add_argument(
		"--vocab-path",
		type=Path,
		default=Path("src/utils/vocab.json"),
		help="Shared vocabulary JSON produced by tokenization.py.",
	)
	parser.add_argument(
		"--glove-path",
		type=Path,
		required=True,
		help="Path to the raw GloVe txt file (e.g., glove.6B.300d.txt).",
	)
	parser.add_argument(
		"--embedding-dim",
		type=int,
		default=None,
		help="Expected embedding dimension. Leave blank to infer from the GloVe file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("models/glove_embeddings.pt"),
		help="Destination .pt file storing the aligned embedding matrix.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Seed for initializing OOV token vectors.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	vocab = load_vocab(args.vocab_path)
	glove_vectors, inferred_dim = load_glove_subset(args.glove_path, vocab, args.embedding_dim)
	matrix, stats = build_embedding_matrix(vocab, glove_vectors, inferred_dim, args.seed)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	save_embeddings(matrix, stats, args.output)

	print(
		f"[GloVe] vocab={stats.vocab_size:,} | dim={stats.embedding_dim} | "
		f"hits={stats.matched_tokens:,} | coverage={stats.token_coverage:.3f}"
	)
	print(f"[GloVe] Embedding matrix saved to {args.output}")


if __name__ == "__main__":
	main()


#example usage: added by @soubhagya:
# python -m src.models.cnn_lstm.GloVe_embeddings \
#   --vocab-path src/utils/vocab.json \
#   --glove-path src/outputfiles/glove.6B.300d.txt \
#   --output models/glove_embeddings.pt