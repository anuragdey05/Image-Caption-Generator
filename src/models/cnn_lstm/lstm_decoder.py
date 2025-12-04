"""LSTM decoder used for caption generation (modified to support pretrained GloVe)."""
from __future__ import annotations
from typing import List
import torch
from torch import nn
from src.embeddings.build_vocab import END_IDX, PAD_IDX, START_IDX


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,       # model internal embedding dim (LSTM input size)
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        image_feat_dim: int = 256,
        glove_dim: int | None = None,   # NEW: set to 300 if using 300-dim GloVe
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_emb_dim = embedding_dim

        # If glove_dim provided and different from embedding_dim, keep embedding at glove_dim
        # and learn a projection glove_dim -> embedding_dim. Otherwise embedding_dim == glove_dim.
        if glove_dim is not None and glove_dim != embedding_dim:
            self.embedding = nn.Embedding(vocab_size, glove_dim, padding_idx=PAD_IDX)
            self.glove_proj = nn.Linear(glove_dim, embedding_dim)
        else:
            # either glove_dim is None, or glove_dim == embedding_dim
            emb_size = embedding_dim if glove_dim is None else glove_dim
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
            self.glove_proj = None  # type: ignore[assignment]

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_init = nn.Linear(image_feat_dim, hidden_dim * num_layers)
        self.cell_init = nn.Linear(image_feat_dim, hidden_dim * num_layers)

    def _init_state(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = image_features.size(0)
        h0 = torch.tanh(self.hidden_init(image_features)).view(batch_size, self.num_layers, self.hidden_dim)
        c0 = torch.tanh(self.cell_init(image_features)).view(batch_size, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = c0.permute(1, 0, 2).contiguous()
        return h0, c0

    def _maybe_project(self, emb: torch.Tensor) -> torch.Tensor:
        # emb shape: (B, T, emb_size) where emb_size == glove_dim (if provided) or embedding_dim
        if getattr(self, "glove_proj", None) is not None:
            return self.glove_proj(emb)
        return emb

    def forward(self, captions_in: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(captions_in)
        embedded = self._maybe_project(embedded)
        hidden, cell = self._init_state(image_features)
        outputs, _ = self.lstm(embedded, (hidden, cell))
        return self.fc(outputs)

    # --- generation paths: apply projection wherever embedding is used ---
    def generate(
        self,
        image_features: torch.Tensor,
        max_length: int = 30,
        greedy: bool = True,
        beam_size: int = 1,
        top_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if beam_size <= 1:
            seq, lengths, scores = self._greedy_decode(image_features, max_length, greedy)
            return seq.unsqueeze(1), lengths.unsqueeze(1), scores.unsqueeze(1)
        seq, lengths, scores = self._beam_search_decode(image_features, max_length, beam_size, top_k)
        return seq, lengths, scores

    def _greedy_decode(
        self,
        image_features: torch.Tensor,
        max_length: int,
        greedy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = image_features.device
        batch_size = image_features.size(0)
        hidden, cell = self._init_state(image_features)
        sequences = torch.full((batch_size, max_length), PAD_IDX, dtype=torch.long, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        scores = torch.zeros(batch_size, dtype=torch.float, device=device)
        inputs = torch.full((batch_size, 1), START_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        last_step = 1

        for step in range(max_length):
            embedded = self.embedding(inputs)              # (B, 1, emb_size)
            embedded = self._maybe_project(embedded)       # (B, 1, embedding_dim)
            outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            logits = self.fc(outputs[:, -1, :])
            if greedy:
                next_token = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            log_probs = torch.log_softmax(logits, dim=-1)
            chosen_scores = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            active_mask = (~finished).float()
            scores = scores + chosen_scores * active_mask

            sequences[:, step] = next_token
            newly_finished = (next_token == END_IDX) & (~finished)
            lengths = torch.where(newly_finished, torch.full_like(lengths, step + 1), lengths)
            finished = finished | (next_token == END_IDX)
            last_step = step + 1
            if finished.all():
                break

            inputs = next_token.unsqueeze(1)
            inputs = torch.where(finished.unsqueeze(1), torch.full_like(inputs, PAD_IDX), inputs)

        lengths = torch.where(lengths == 0, torch.full_like(lengths, last_step), lengths)
        return sequences, lengths, scores

    def _beam_search_decode(
        self,
        image_features: torch.Tensor,
        max_length: int,
        beam_size: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = image_features.device
        batch_size = image_features.size(0)
        top_k = max(1, min(top_k, beam_size))

        batch_seqs: List[torch.Tensor] = []
        batch_lengths: List[torch.Tensor] = []
        batch_scores: List[torch.Tensor] = []

        for b in range(batch_size):
            single_feat = image_features[b : b + 1]
            hidden, cell = self._init_state(single_feat)
            beam = [
                {
                    "seq": torch.tensor([START_IDX], device=device, dtype=torch.long),
                    "score": 0.0,
                    "hidden": hidden.detach().clone(),
                    "cell": cell.detach().clone(),
                }
            ]
            completed = []

            for _ in range(max_length):
                new_beam = []
                for candidate in beam:
                    last_token = candidate["seq"][-1].item()
                    if last_token == END_IDX:
                        completed.append(candidate)
                        continue
                    input_token = torch.tensor([[last_token]], device=device, dtype=torch.long)
                    embedded = self.embedding(input_token)
                    embedded = self._maybe_project(embedded)
                    outputs, (h_next, c_next) = self.lstm(embedded, (candidate["hidden"], candidate["cell"]))
                    logits = self.fc(outputs[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    top_scores, top_idx = torch.topk(log_probs, beam_size)
                    for score, idx in zip(top_scores, top_idx):
                        new_seq = torch.cat([candidate["seq"], idx.view(1)])
                        new_beam.append(
                            {
                                "seq": new_seq,
                                "score": candidate["score"] + float(score.item()),
                                "hidden": h_next.detach().clone(),
                                "cell": c_next.detach().clone(),
                            }
                        )
                if not new_beam:
                    break
                new_beam.sort(key=lambda item: item["score"], reverse=True)
                beam = new_beam[:beam_size]

            completed.extend(beam)
            completed.sort(key=lambda item: item["score"], reverse=True)
            top_candidates = completed[:top_k]
            if not top_candidates:
                top_candidates = beam[:1]

            seq_tensor = torch.full((top_k, max_length), PAD_IDX, dtype=torch.long, device=device)
            len_tensor = torch.zeros(top_k, dtype=torch.long, device=device)
            score_tensor = torch.full((top_k,), float("-inf"), dtype=torch.float, device=device)

            for i, cand in enumerate(top_candidates):
                seq_tokens = cand["seq"][1 : max_length + 1]
                length = min(len(seq_tokens), max_length)
                if length > 0:
                    seq_tensor[i, :length] = seq_tokens[:length]
                len_tensor[i] = length
                score_tensor[i] = cand["score"]

            batch_seqs.append(seq_tensor)
            batch_lengths.append(len_tensor)
            batch_scores.append(score_tensor)

        return (
            torch.stack(batch_seqs, dim=0),
            torch.stack(batch_lengths, dim=0),
            torch.stack(batch_scores, dim=0),
        )
