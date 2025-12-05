# Image-Caption-Generator Project

This repo prepares subsets of the **ArtEmis** annotations and **WikiArt** images and trains two captioning models:

- A **CNN + LSTM** baseline.
- A **Vision Transformer (ViT) + Transformer Decoder** model.

The code is organized so that **data preparation**, **tokenization / vocab**, and **embeddings** are shared between both models.

---

## 1. Project Layout

```text
data/
  artemis_sample.csv               # stratified ArtEmis subset used for training
  wikiart/                         # full WikiArt images (unzipped)
  wikiart_sample/                  # sampled subset (per style)
  wikiart_sample_128/              # resized 128×128 images for training
  official_data/
    artemis_dataset_release_v0.csv # original ArtEmis CSV

src/
  pre_processing.py                # end-to-end sampling + image copying + resizing

  utils/
    tokenization.py                # shared tokenizer + vocab build/save/load
    image_utils.py                 # shared ViT-style image preprocessing
    vocab.json                     # saved vocabulary (tokens ↔ indices)

  datasets/
    artemis_dataset.py             # Dataset for (image, caption) pairs

  models/
    cnn_lstm/
      cnn_encoder.py               # CNN encoder backbone for images
      lstm_decoder.py              # LSTM-based caption decoder
      cnn_lstm_model.py            # Combined CNN+LSTM caption model
      GloVe_embeddings.py          # builds aligned GloVe embedding matrix
      fasttest_embeddings.py       # (typo) FastText embedding builder
      tf_idf_embeddings.py         # TF‑IDF-style embeddings for vocab

    vit/
      vision_encoder_vit.py        # ViT-style patch encoder for images
      text_decoder.py              # Transformer text decoder with cross-attention
      caption_transformer.py       # Wraps ViT encoder + text decoder
      config_transformer.py        # Shared hyperparameters for ViT + decoder

  training/
    train_cnn_lstm.py              # training script for CNN+LSTM model
    train_vit.py                   # training script for ViT-based model

outputfiles/
  vit_epochXXX_valY.YYY.pt         # saved ViT model checkpoints
  glove_embeddings.pt              # GloVe embedding matrix aligned to vocab
  fasttext_embeddings.pt           # FastText embedding matrix
  tfidf_embeddings.pt              # TF‑IDF embedding matrix
```

---

## 2. End-to-End Workflow

### 2.1 Data Preparation

Script: [`src/pre_processing.py`](src/pre_processing.py) (`DataPreparator`)

Steps:

1. **Unzip WikiArt (optional)**  
   - Input: `data/wikiart.zip`  
   - Output: `data/wikiart/<art_style>/<image>.jpg`

2. **Create a stratified ArtEmis sample**  
   - Input: [`data/official_data/artemis_dataset_release_v0.csv`](data/official_data/artemis_dataset_release_v0.csv)  
   - Output: [`data/artemis_sample.csv`](data/artemis_sample.csv)  
   - Stratifies by **art_style** and emotion so that your subset covers styles/emotions reasonably well.

3. **Copy corresponding WikiArt images**  
   - Copies images used in `artemis_sample.csv` from `data/wikiart/` into [`data/wikiart_sample/`](data/wikiart_sample/).

4. **Resize images**  
   - Resizes copies to **128×128** and writes them to [`data/wikiart_sample_128`](data/wikiart_sample_128).  
   - These 128×128 images are what both models train on.

Run:

```bash
python src/pre_processing.py
```

---

### 2.2 Vocabulary and Tokenization

File: [`src/utils/tokenization.py`](src/utils/tokenization.py)

Core class: `Tokenizer`

- **Goal**: One shared text pipeline and vocabulary for:
  - CNN + LSTM decoder.
  - ViT + Transformer decoder.
  - External embeddings (GloVe, FastText, TF‑IDF).

Main responsibilities:

- **Build vocab from ArtEmis captions**  
  - Reads `utterance` column from [`data/artemis_sample.csv`](data/artemis_sample.csv).  
  - Assigns IDs to:
    - PAD token
    - UNK (unknown) token
    - BOS (begin-of-sentence)
    - EOS (end-of-sentence)
  - Builds `word2idx` / `idx2word` mappings.
  - Saves to [`src/utils/vocab.json`](src/utils/vocab.json) so all models share identical indices.

- **Tokenization + encoding**
  - Normalizes text (lowercase, strips punctuation, collapses whitespace).
  - Splits into tokens.
  - `encode_caption(text, max_len)`:
    - `[BOS] tokens [EOS]`
    - Pads or truncates to `max_len`.
    - Returns a list of token IDs.

- **Decoding**
  - `decode_tokens(ids)`:
    - Stops at EOS.
    - Skips PAD.
    - Returns the readable token list.

Usage to build vocab from ArtEmis:

```bash
python src/utils/tokenization.py
# (builds vocab from data/artemis_sample.csv and writes src/utils/vocab.json)
```

---

### 2.3 Dataset for Training

File: [`src/datasets/artemis_dataset.py`](src/datasets/artemis_dataset.py)

Class: `ArtemisDataset`

- Inputs:
  - `csv_path`: usually `data/artemis_sample.csv`.
  - `img_root`: `data/wikiart_sample_128`.
  - `tokenizer`: shared `Tokenizer` instance.
  - `max_len`: max token sequence length (e.g., 32).

- Per sample:
  - Builds image path:  
    `img_root / art_style / (painting + ".jpg")`
  - Uses [`src/utils/image_utils.py`](src/utils/image_utils.py) (`load_image`) to:
    - Load image.
    - Resize to 128×128 (if needed).
    - Normalize for ViT: mean = std = 0.5.
    - Returns `image_tensor` of shape `(3, 128, 128)`.

  - Tokenizes caption from `utterance`:
    - `caption_ids = tokenizer.encode_caption(caption, max_len)`.
    - `caption_in = caption_ids[:-1]` → input to decoder (BOS … token_{T-1}).
    - `caption_out = caption_ids[1:]` → training target (token_1 … EOS).

- Output:
  - `image_tensor`, `caption_in`, `caption_out`

This dataset is used by both CNN+LSTM and ViT training scripts (directly or via model-specific collate functions).

---

## 3. Models

### 3.1 CNN + LSTM Baseline

Folder: [`src/models/cnn_lstm`](src/models/cnn_lstm/__init__.py)  
Script: [`src/training/train_cnn_lstm.py`](src/training/train_cnn_lstm.py)

**Architecture**

- [`CNNEncoder`](src/models/cnn_lstm/cnn_encoder.py)
  - Convolutional backbone (typically a ResNet-type encoder).
  - Produces a fixed-size image feature vector of dimension `image_feature_dim` (default: 256).

- [`LSTMDecoder`](src/models/cnn_lstm/lstm_decoder.py)
  - Inputs:
    - `caption_in` sequence of token IDs.
    - Image feature vector from the encoder.
  - Embeds tokens (optionally with pre-trained embeddings).
  - Concatenates or conditions on image features (e.g., via initial hidden state).
  - Uses an LSTM to produce hidden states.
  - Outputs logits over vocabulary for each time step.

- [`ImageCaptioningCNNLSTM`](src/models/cnn_lstm/cnn_lstm_model.py)
  - Wraps encoder + decoder into a single `nn.Module`.
  - Methods:
    - `forward(images, captions_in) -> logits`
    - `generate(images, ...) -> sampled token sequences` for inference.

**Embeddings options**

- [`GloVe_embeddings.py`](src/models/cnn_lstm/GloVe_embeddings.py):
  - Aligns GloVe vectors with the shared vocab in [`src/utils/vocab.json`](src/utils/vocab.json).
  - Saves aligned matrix and coverage stats to [`src/outputfiles/glove_embeddings.pt`](src/outputfiles/glove_embeddings.pt).

- [`fasttest_embeddings.py`](src/models/cnn_lstm/fasttest_embeddings.py):
  - Aligns **FastText** vectors (binary `.bin`) with vocab.
  - Saves matrix to [`src/outputfiles/fasttext_embeddings.pt`](src/outputfiles/fasttext_embeddings.pt).

- [`tf_idf_embeddings.py`](src/models/cnn_lstm/tf_idf_embeddings.py):
  - Computes TF‑IDF-style embeddings for each vocab token using ArtEmis captions.
  - Saves matrix to [`src/outputfiles/tfidf_embeddings.pt`](src/outputfiles/tfidf_embeddings.pt).

You can load any of these `.pt` files and copy their matrices into an `nn.Embedding` layer of the LSTM decoder.

---

### 3.2 Vision Transformer (ViT) + Transformer Decoder

Folder: [`src/models/vit`](src/models/vit/__init__.py)  
Script: [`src/training/train_vit.py`](src/training/train_vit.py)

#### 3.2.1 Vision Encoder (ViT)

File: [`src/models/vit/vision_encoder_vit.py`](src/models/vit/vision_encoder_vit.py)

Config: `ViTConfig`:

- `image_size`: 128
- `patch_size`: 16
- `d_model`: 256
- `num_layers`: 4 (for the encoder)
- `num_heads`: 8
- `mlp_ratio`: 4.0
- `dropout`: 0.1
- `attn_dropout`: 0.0

Components:

- `PatchEmbedding`
  - Uses a `Conv2d` with `kernel_size = patch_size` and `stride = patch_size`.
  - Turns `(B, 3, 128, 128)` into `(B, num_patches, d_model)` where  
    `num_patches = (image_size / patch_size)^2 = (128 / 16)^2 = 64`.

- Positional Embeddings
  - Learnable positional embeddings `pos_embedding` of shape `(1, num_patches, d_model)`.

- `TransformerEncoderBlock` (stacked `num_layers` times)
  - LayerNorm → Multi-Head Self-Attention → residual.
  - LayerNorm → Feed-Forward Network (MLP) → residual.
  - Dropout inside attention and MLP.

Output:

- `VisionEncoderViT.forward(images)` returns `image_tokens` of shape `(B, num_patches, d_model)`.

These tokens are used as **key/value** for cross-attention in the text decoder.

#### 3.2.2 Text Transformer Decoder

File: [`src/models/vit/text_decoder.py`](src/models/vit/text_decoder.py)

Config: `TextDecoderConfig`:

- `vocab_size`
- `max_length`: maximum caption length (e.g., 32).
- `d_model`: 256
- `num_heads`: 8
- `num_layers`: 4
- `mlp_ratio`: 4.0
- `dropout`: 0.1
- `attn_dropout`: 0.0
- `pad_idx`: index of PAD token.

Components:

- Token embedding: `nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)`.
- Positional embedding: learned or sinusoidal positions up to `max_length`.
- `DecoderBlock` × `num_layers`:
  - Masked self-attention:
    - Causal mask to prevent attending to future tokens.
    - Optional padding mask for short sequences.
  - Cross-attention:
    - Attends over `image_tokens` from ViT encoder.
  - Feed-forward network.
  - Each sub-layer uses LayerNorm + residual connection.
- Final `nn.LayerNorm` + linear projection → logits over vocabulary per time step.

Methods:

- `forward(captions_in, memory) -> logits`:
  - `captions_in`: `(B, T)` (without EOS).
  - `memory`: `(B, num_patches, d_model)` from the vision encoder.
  - Returns `logits` `(B, T, vocab_size)`.

- `generate(memory, bos_token_id, eos_token_id, max_length)`:
  - Autoregressive greedy decoding.
  - Repeatedly calls the decoder with growing prefix until EOS or max length.

#### 3.2.3 Caption Transformer Wrapper

File: [`src/models/vit/caption_transformer.py`](src/models/vit/caption_transformer.py)

Class: `CaptionTransformer`

- Composes:
  - `VisionEncoderViT` (image → tokens).
  - `TextTransformerDecoder` (tokens + captions_in → logits).

- `__init__(vision_config, decoder_config)`:
  - Uses [`TransformerHyperParams`](src/models/vit/config_transformer.py) to generate consistent configs.

- `forward(images, captions_in)`:
  - `memory = vision_encoder(images)`
  - `logits = text_decoder(captions_in, memory)`
  - Used during training with teacher forcing.

- `generate(images, bos_id, eos_id, max_length)`:
  - Encodes images once.
  - Runs decoder’s `generate` for inference.

---

## 4. Hyperparameters and Config

File: [`src/models/vit/config_transformer.py`](src/models/vit/config_transformer.py)

Class: `TransformerHyperParams`

- Centralizes hyperparameters:

  - `vocab_size`
  - `pad_idx`
  - `d_model = 256`
  - `num_heads = 8`
  - `num_layers = 4`
  - `patch_size = 16`
  - `max_seq_len = 32`
  - `image_size = 128`
  - `dropout = 0.1`
  - `attn_dropout = 0.0`
  - `mlp_ratio = 4.0`

- Methods:
  - `.vision_config()` → `ViTConfig` for the encoder.
  - `.decoder_config()` → `TextDecoderConfig` for the text decoder.

This ensures encoder and decoder always share the same `d_model`, `num_heads`, `num_layers`, and sequence length limits.

---

## 5. Training Scripts

### 5.1 CNN + LSTM Training

Script: [`src/training/train_cnn_lstm.py`](src/training/train_cnn_lstm.py)

Key steps:

1. **Argument parsing**: path to CSV, images root, vocab path, hyperparameters (embedding dim, hidden dim, etc.).
2. **Reproducibility**: `seed_everything(seed)`.
3. **Vocabulary**:
   - `build_or_load_vocab(csv_path, vocab_path, min_freq)`.
   - Uses a `Vocabulary` (similar to `Tokenizer`) and saves to JSON.

4. **Dataset + DataLoader**:
   - Uses a dataset that yields `(image, caption_in, caption_out)`.
   - Collate function pads caption sequences.

5. **Model**:
   - `ImageCaptioningCNNLSTM` with:
     - `image_feature_dim` (default 256)
     - `embedding_dim` (default 256)
     - `hidden_dim` (default 256)
     - `num_layers` (default 1)
     - `dropout` (default 0.1)

6. **Loss + Optimizer**:
   - `nn.CrossEntropyLoss(ignore_index=PAD_IDX)` → categorical cross-entropy over next token.
   - `Adam` optimizer.

7. **Training loop**:
   - Mixed precision (`GradScaler` + `autocast`) when on CUDA.
   - Gradient clipping (`max_grad_norm`).
   - Scheduler: `ReduceLROnPlateau` on validation loss (`factor`, `patience`).
   - Early stopping: `EarlyStopping(patience)` on validation loss.
   - Logs:
     - Training + validation loss per epoch.
     - LR and gradient norms.
     - Writes CSV log file.

8. **Checkpoints**:
   - Saves best model states (by validation loss) to a configurable output directory (e.g., `models/`).

---

### 5.2 ViT + Transformer Decoder Training

Script: [`src/training/train_vit.py`](src/training/train_vit.py)

Model: `CaptionTransformer` (ViT encoder + transformer decoder)

Key points:

1. **Arguments**:
   - `--csv`: `data/artemis_sample.csv` (default).
   - `--images-root`: `data/wikiart_sample_128`.
   - `--vocab-path`: `src/utils/vocab.json`.
   - `--output-dir`: `src/outputfiles` (checkpoints stored here).
   - `--batch-size`, `--epochs`, `--val-ratio`, `--num-workers`.
   - `--lr`, `--weight-decay`.
   - `--d-model`, `--num-heads`, `--num-layers`, `--patch-size`, `--max-len`.
   - Early stopping and scheduler parameters:
     - `--early-stop-patience`
     - `--scheduler-factor`
     - `--scheduler-patience`
   - `--max-grad-norm`, `--no-amp`, `--device`, `--seed`, `--log-file`.

2. **Setup**:
   - Seeds all RNGs (`seed_everything(seed)`).
   - Loads `Tokenizer` from vocab JSON.
   - Builds `TransformerHyperParams` with `vocab_size`, `pad_idx`, and model dimensions.

3. **Dataset + Dataloaders**:
   - Uses [`ArtemisDataset`](src/datasets/artemis_dataset.py) to produce `(image_tensor, caption_in, caption_out)`.
   - Splits into train/val using `val_ratio`.
   - `collate_batch` stacks image and caption tensors.

4. **Model**:
   - `CaptionTransformer(vision_config=hyper.vision_config(), decoder_config=hyper.decoder_config())`.
   - Moves model to `device`.

5. **Loss + Optimizer + Scheduler**:
   - `nn.CrossEntropyLoss(ignore_index=pad_idx)`  
     → categorical cross-entropy to predict `caption_out` tokens.
   - `AdamW` optimizer with weight decay.
   - `ReduceLROnPlateau` scheduler on validation loss.

6. **Mixed Precision + Grad Clipping**:
   - `GradScaler` + `autocast` (if CUDA and `--no-amp` not set).
   - `clip_grad_norm_` with `max_grad_norm`.

7. **Training loop**:
   - `train_one_epoch`:
     - For each batch:
       - Forward: `logits = model(images, captions_in)`.
       - Loss: `CrossEntropy` between `logits.view(-1, vocab_size)` and `captions_out.view(-1)`.
       - Backprop (scaled).
       - Gradient clipping.
       - Optimizer + scaler step.
       - Logs batch loss periodically (`--log-interval`).
     - Returns average training loss and average gradient norm.

   - `evaluate`:
     - Runs model in eval mode and computes average loss on val set.

   - After each epoch:
     - Step the scheduler with `val_loss`.
     - Append to CSV log: epoch, train_loss, val_loss, learning rate, average grad norm.
     - Save checkpoint on best validation loss (into [`src/outputfiles`](src/outputfiles)):
       - Model state.
       - Optimizer state.
       - Scaler state.
       - Hyperparameters.
       - Training history.

   - Early stopping:
     - Stops training if validation loss does not improve for `early_stop_patience` epochs.

8. **Objective**:
   - Like the CNN+LSTM model, uses **categorical cross-entropy** to predict the next token in sequence (`caption_out`) given:
     - Image tokens from ViT.
     - Past caption tokens (`caption_in`).

9. **End-to-end training**:
   - Both **visual** (ViT) and **text** (decoder embeddings, attention layers) parameters are trained jointly.

Example command:

```bash
python -m src.training.train_vit \
  --csv data/artemis_sample.csv \
  --images-root data/wikiart_sample_128 \
  --vocab-path src/utils/vocab.json \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-4
```

Checkpoints will appear under [`src/outputfiles`](src/outputfiles), e.g.:

```text
src/outputfiles/vit_epoch010_val5.009.pt
```

These `.pt` files contain the model state and training metadata.

---

## 6. Summary of Design Choices

- **Shared vocabulary / tokenizer**:
  - A single `Tokenizer` and `vocab.json` are used across:
    - CNN+LSTM decoder.
    - ViT+Transformer decoder.
    - GloVe/FastText/TF‑IDF embeddings.
  - Guarantees consistency of token indices.

- **Image resolution**:
  - All models train on `128×128` RGB images to:
    - Reduce compute.
    - Simplify ViT patching (`128 / 16 = 8`, giving 64 patches).

- **Transformer hyperparameters** (default via `TransformerHyperParams`):
  - `d_model = 256`, `num_heads = 8`, `num_layers = 4`, `mlp_ratio = 4.0`.
  - `patch_size = 16`, `max_seq_len = 32`.
  - Chosen as a compromise between capacity and small-batch training stability on ArtEmis subset.

- **Optimization**:
  - **CNN+LSTM**: `Adam`, standard cross-entropy, gradient clipping, ReduceLROnPlateau, early stopping.
  - **ViT**: `AdamW` with weight decay, same scheduler and early stopping, plus mixed precision to support higher batch sizes.

- **Logging & Reproducibility**:
  - Deterministic seeding of Python, NumPy (where used), and PyTorch.
  - CSV logs with epoch-wise metrics.
  - Checkpoints store hyperparameters and histories to reproduce runs.

---

## 7. Next Steps / Inference

- To run inference with either model:
  - Load the checkpoint (`torch.load`).
  - Rebuild the model with the same hyperparameters.
  - Load `state_dict`.
  - Use:
    - `ImageCaptioningCNNLSTM.generate(images, ...)` for the CNN+LSTM baseline.
    - `CaptionTransformer.generate(images, bos_id, eos_id, max_length)` for the ViT model.
- Use `Tokenizer.decode_tokens` to convert output token IDs back into human-readable captions.

This README should give you a consistent view of how the entire pipeline fits together: data, tokenization, models, training, and saved artifacts.

