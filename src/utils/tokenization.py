"""Shared caption tokenization utilities for CNN+LSTM and ViT models."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

_SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
_PUNCT_PATTERN = re.compile(r"[^a-z0-9' ]+")


class Tokenizer:
	"""Shared tokenizer with vocab + encode/decode helpers."""

	PAD_TOKEN = PAD_TOKEN
	UNK_TOKEN = UNK_TOKEN
	BOS_TOKEN = BOS_TOKEN
	EOS_TOKEN = EOS_TOKEN

	def __init__(self, word2idx: Dict[str, int], idx2word: Dict[int, str]):
		if not word2idx or not idx2word:
			raise ValueError("Tokenizer requires word2idx and idx2word mappings")
		self.word2idx = word2idx
		self.idx2word = idx2word
		self.pad_idx = self.word2idx[self.PAD_TOKEN]
		self.unk_idx = self.word2idx[self.UNK_TOKEN]
		self.bos_idx = self.word2idx[self.BOS_TOKEN]
		self.eos_idx = self.word2idx[self.EOS_TOKEN]

	@staticmethod
	def clean_text(text: str) -> str:
		"""Lowercase, strip punctuation, collapse whitespace."""
		text = text.lower()
		text = _PUNCT_PATTERN.sub(" ", text)
		return re.sub(r"\s+", " ", text).strip()

	@staticmethod
	def _tokenize(text: str) -> List[str]:
		"""Whitespace tokenize already-clean text."""
		return [tok for tok in text.split(" ") if tok]

	@classmethod
	def build_vocab(
		cls,
		captions: Iterable[str],
		vocab_size: int,
		pretokenized: Optional[Iterable[List[str]]] = None,
	) -> "Tokenizer":
		"""Create vocab mappings and return a Tokenizer instance."""
		if vocab_size < len(_SPECIAL_TOKENS):
			raise ValueError("vocab_size must be >= number of special tokens")

		if pretokenized is None:
			pretokenized = [cls._tokenize(cls.clean_text(c)) for c in captions]
		else:
			pretokenized = list(pretokenized)

		counter: Counter[str] = Counter()
		for tokens in pretokenized:
			counter.update(tokens)

		most_common = counter.most_common(max(0, vocab_size - len(_SPECIAL_TOKENS)))
		vocab_words = [token for token, _ in most_common]

		word2idx: Dict[str, int] = {}
		idx2word: Dict[int, str] = {}

		def _add(token: str):
			idx = len(word2idx)
			word2idx[token] = idx
			idx2word[idx] = token

		for token in _SPECIAL_TOKENS:
			_add(token)
		for token in vocab_words:
			if token not in word2idx:
				_add(token)

		return cls(word2idx, idx2word)

	@classmethod
	def vocab_size_for_coverage(
		cls,
		captions: Iterable[str],
		coverage: float = 0.97,
		pretokenized: Optional[Iterable[List[str]]] = None,
	) -> int:
		"""Return vocab size (incl. specials) covering given fraction of tokens."""
		if not 0 < coverage <= 1:
			raise ValueError("coverage must be in (0, 1]")

		if pretokenized is None:
			pretokenized = [cls._tokenize(cls.clean_text(c)) for c in captions]
		else:
			pretokenized = list(pretokenized)

		counter: Counter[str] = Counter()
		for tokens in pretokenized:
			counter.update(tokens)

		if not counter:
			return len(_SPECIAL_TOKENS)

		total_tokens = sum(counter.values())
		target = total_tokens * coverage
		accum = 0
		kept = 0
		for _, freq in counter.most_common():
			accum += freq
			kept += 1
			if accum >= target:
				break

		return kept + len(_SPECIAL_TOKENS)

	def encode_caption(self, text: str, max_len: int) -> List[int]:
		"""Convert caption to fixed-length IDs (BOS..EOS, pad/truncate)."""
		if max_len < 2:
			raise ValueError("max_len must be >= 2 to fit BOS/EOS")

		tokens = self._tokenize(self.clean_text(text))
		seq = [self.bos_idx]
		seq.extend(self.word2idx.get(tok, self.unk_idx) for tok in tokens)
		seq.append(self.eos_idx)

		if len(seq) < max_len:
			seq.extend([self.pad_idx] * (max_len - len(seq)))
		else:
			seq = seq[:max_len]
			seq[-1] = self.eos_idx
		return seq

	def decode_tokens(self, token_ids: Sequence[int]) -> List[str]:
		"""Convert token IDs back to tokens, stopping at EOS and skipping PAD."""
		decoded: List[str] = []
		for idx in token_ids:
			if idx == self.pad_idx:
				continue
			if idx == self.eos_idx:
				break
			token = self.idx2word.get(idx)
			if token is not None and token not in _SPECIAL_TOKENS:
				decoded.append(token)
		return decoded

	def save(self, filepath: Path | str) -> None:
		"""Persist vocab so both models share identical mappings."""
		path = Path(filepath)
		payload = {
			"word2idx": self.word2idx,
			"idx2word": {str(k): v for k, v in self.idx2word.items()},
		}
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	@classmethod
	def load(cls, filepath: Path | str) -> "Tokenizer":
		"""Load vocab saved via save()."""
		payload = json.loads(Path(filepath).read_text(encoding="utf-8"))
		word2idx = {k: int(v) for k, v in payload["word2idx"].items()}
		idx2word = {int(k): v for k, v in payload["idx2word"].items()}
		return cls(word2idx, idx2word)


if __name__ == "__main__":
	# Use the stratified ArtEmis sample to build the shared vocab
	project_root = Path(__file__).resolve().parents[2]
	csv_path = project_root / "data" / "artemis_sample.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Expected ArtEmis sample at {csv_path}")

	df = pd.read_csv(csv_path)
	if "utterance" not in df.columns:
		raise ValueError("artemis_sample.csv must contain an 'utterance' column")

	captions = df["utterance"].dropna().astype(str).tolist()
	print(f"Building vocab from {len(captions)} ArtEmis captions...")
	tokenized_captions = [Tokenizer._tokenize(Tokenizer.clean_text(c)) for c in captions]
	vocab_size = Tokenizer.vocab_size_for_coverage(
		captions,
		coverage=0.97,
		pretokenized=tokenized_captions,
	)
	print(f"97% coverage vocab size: {vocab_size}")
	tokenizer = Tokenizer.build_vocab(captions, vocab_size=vocab_size, pretokenized=tokenized_captions)

	out_path = Path(__file__).with_name("vocab.json")
	tokenizer.save(out_path)
	print(f"Saved vocab entries: {len(tokenizer.word2idx)} -> {out_path}")

	sample = captions[0] if captions else "a painting."
	encoded = tokenizer.encode_caption(sample, max_len=32)
	decoded = tokenizer.decode_tokens(encoded)
	print("Sample text:", sample)
	print("Encoded:", encoded)
	print("Decoded tokens:", decoded)

