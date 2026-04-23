# Leon Clip

Tri-modal CLIP-style training for H3-anchored text, image, and graph embeddings.

This stack now supports robust multi-positive contrastive learning, hard-negative
weighting, modality dropout, learnable missing tokens, stronger projection heads,
and retrieval-focused evaluation.

## Project Structure

- `triple_encoder/store.py` - SQLite-backed ingestion and retrieval.
- `triple_encoder/dataloader.py` - H3 triple construction and batch sampling.
- `triple_encoder/model.py` - projection heads and shared embedding space.
- `triple_encoder/losses.py` - multi-positive objectives and hard-negative weighting.
- `triple_encoder/eval.py` - retrieval metrics for all modality directions.
- `train.py` - CLI training entrypoint.
- `tests/` - lightweight correctness tests.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

The training code expects an embeddings root with these subdirectories:

- `text_embeddings/` - parquet files with H3 ids and text embeddings.
- `image_embeddings/` - pickle files with image embeddings.
- `graph_embeddings/` - pickle files with graph embeddings.

The ingester scans those folders recursively at the top level and builds a SQLite store at `--store-db`.

## Ingestion

`EmbeddingStore.build_from_embeddings_root()` validates each row before insertion:

- invalid H3 cells are skipped with a warning,
- rows whose vector dimension does not match the first valid vector for that modality are skipped with a warning,
- repeated builds are idempotent because exact duplicate vectors are deduplicated with a stable hash.

Multiple embeddings per `(modality, h3)` are supported intentionally:

- distinct vectors for the same modality and H3 cell are preserved,
- exact duplicate vectors are ignored,
- uniqueness is enforced on `(modality, h3, vec_hash)`.

If `--rebuild-store` is passed, the SQLite tables are cleared first.

## Target H3 Resolutions

The dataset anchors source cells to the requested target resolutions. If a source cell is already at one of the target resolutions, it is kept as-is. Otherwise the nearest lower target resolution is used. Cells that cannot be anchored into the target set are skipped.

## Hierarchical Features

Text and graph features are hierarchical. For each H3 cell, the dataset builds a tensor by concatenating:

1. parent embedding,
2. center embedding,
3. pooled child embedding.

Image features are intentionally non-hierarchical. The dataset uses the exact center-cell image embedding only.

The dataset also emits batch metadata used by the loss/eval stack:

- `h3`
- `h3_resolution`
- precomputed ancestors such as `h3_parent_7`

To improve throughput, dataset metadata is cached up front:

- H3 resolution per sample,
- parent/child relationships per sample,
- configured ancestor ids,
- modality presence maps per sample.

Embedding vectors are fetched lazily and cached per worker to avoid repeated SQLite queries for the same `(modality, h3)` during training.

## Presence Masks

Presence masks describe the actual tensor that is returned:

- `image_present` is true only when the exact image embedding exists for the center cell.
- `text_present` and `graph_present` are true when the hierarchical tensor contains at least one real embedding from the parent, center, or child cells.

This keeps the masks aligned with the tensors fed into the loss.

During training, modality dropout produces effective presence masks per batch.
Raw dataset presence is preserved as `*_present_raw` and never overwritten in storage.

## Multi-Positive Loss

Loss mode is configurable:

- `single_positive`: diagonal-only baseline
- `multi_positive`: multiple positives per anchor using an `N x N` positive mask

Positive definitions:

- exact same H3 anchor
- optional cross-resolution positives using:
  - `exact`
  - `parent_child`
  - `shared_ancestor`

Cross-resolution positives are excluded from negatives automatically.

## Hard Negatives

Hard negatives are batch-local and configurable via H3 relationships:

- `same_resolution`
- `shared_parent`
- `k_ring`
- `combined`

Hard negatives are upweighted in the denominator by `--hard-negative-weight`.

## Presence-Aware Normalization

Pair losses use valid anchors only (rows with at least one valid positive).

Modes:

- `mean_valid`: mean over valid anchors (default)
- `proportional`: mean over valid anchors scaled by valid-anchor ratio
- `off`: no presence-aware adjustment

Logged per pair:

- valid anchors
- valid positives
- multi-positive matches
- hard-negative counts

## Projection Heads and Missing Tokens

Projection heads are configurable with `--head-type`:

- `linear`
- `mlp`
- `residual_mlp` (default)

Learned missing tokens are optional (`--use-learned-missing-tokens`, on by default)
and are inserted before projection when a modality is missing or dropped.

## Retrieval Evaluation

Evaluation computes retrieval quality for each pair and direction:

- graph -> text and text -> graph
- graph -> image and image -> graph
- text -> image and image -> text

Metrics:

- Recall@1
- Recall@5
- Recall@10
- median rank
- MRR

Evaluation uses the same positive mask rules as training.

Evaluation is intentionally configurable because retrieval metrics can be expensive on large validation splits:

- `--eval-every` controls step-level eval frequency,
- `--max-eval-samples` bounds eval set size strictly,
- `--eval-at-epoch-end` enables optional end-of-epoch eval.

## Sampler Behavior

`ImageStratifiedBatchSampler` visits each example at most once per epoch. It shuffles image and non-image pools independently, then distributes image examples across the epoch as evenly as possible so each batch reaches the requested minimum image ratio when enough image samples exist.

This is a coverage-oriented sampler, not a resampling sampler.

## Training

### Baseline mode (ablations)

```bash
python train.py \
  --training-mode baseline \
  --embeddings-root data \
  --store-db data/embeddings/embeddings.sqlite \
  --rebuild-store \
  --disable-wandb
```

### Improved default mode

```bash
python train.py \
  --training-mode improved \
  --embeddings-root data \
  --store-db data/embeddings/embeddings.sqlite \
  --rebuild-store \
  --eval-every 200 \
  --val-split 0.1 \
  --eval-use-ema \
  --disable-wandb
```

### EMA-focused evaluation run

```bash
python train.py \
  --embeddings-root data \
  --store-db data/embeddings/embeddings.sqlite \
  --eval-use-ema \
  --use-ema \
  --ema-decay 0.999 \
  --disable-wandb
```

Use `--target-resolutions 7,8,9` to control which H3 levels are included.

Recommended quality-first run:

```bash
python train.py \
  --training-mode improved \
  --loss-mode multi_positive \
  --hard-negative-mode combined \
  --allow-cross-resolution-positives \
  --cross-resolution-mode shared_ancestor \
  --cross-resolution-ancestor-res 7 \
  --head-type residual_mlp \
  --head-hidden-dim 2048 \
  --use-learned-missing-tokens \
  --use-ema \
  --eval-use-ema \
  --batch-size 128 \
  --num-workers 4
```

Recommended fast-safe throughput settings (without removing major features):

```bash
python train.py \
  --training-mode improved \
  --num-workers 4 \
  --eval-every 400 \
  --max-eval-samples 1024 \
  --disable-wandb
```

## Key CLI flags

- `--loss-mode {single_positive,multi_positive}`
- `--allow-cross-resolution-positives` / `--disable-cross-resolution-positives`
- `--cross-resolution-mode {exact,parent_child,shared_ancestor}`
- `--cross-resolution-ancestor-res`
- `--hard-negative-mode {none,same_resolution,shared_parent,k_ring,combined}`
- `--hard-negative-weight`
- `--hard-negative-radius`
- `--hard-negative-parent-res-delta`
- `--text-modality-dropout`
- `--image-modality-dropout`
- `--graph-modality-dropout`
- `--presence-normalization {off,mean_valid,proportional}`
- `--head-type {linear,mlp,residual_mlp}`
- `--head-hidden-dim`
- `--head-dropout`
- `--head-residual-layers`
- `--use-learned-missing-tokens`
- `--lr`
- `--logit-scale-lr`
- `--weight-decay`
- `--warmup-steps`
- `--warmup-ratio`
- `--grad-clip-norm`
- `--use-ema` / `--disable-ema`
- `--ema-decay`
- `--eval-every`
- `--eval-at-epoch-end`
- `--val-split`
- `--max-eval-samples`
- `--eval-use-ema`
- `--log-branch-norms`

## Tests

```bash
pytest
```

## Known Limitations

- Hard negatives are batch-local only (no global mining).
- Retrieval evaluation is brute-force batch aggregation and can be memory intensive for very large validation sets.
- Image features remain non-hierarchical.