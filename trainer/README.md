# NNUE Training CLI Steps

This README is only the command-line workflow to train an NNUE model.

**Working directory:** run every command from the **repository root** (`chess-game/`, the directory that contains `trainer/`). Data paths below use `trainer/localdata/` as in the default layout.

**macOS:** long-running steps below are prefixed with `caffeinate -i` so the machine does not sleep while the process is idle. On Linux or Windows, omit `caffeinate -i` and run the `python3 ...` command only.

## 0) Install dependencies

```bash
pip install torch zstandard pyyaml python-chess numpy
```

## 1) Split raw eval file into train/val

If you only have one large file (example: `trainer/localdata/lichess_db_eval.jsonl.zst`), split it:

```bash
caffeinate -i python3 trainer/split_data.py \
  --input trainer/localdata/lichess_db_eval.jsonl.zst \
  --val-size 1000000 \
  --val-output trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --train-output trainer/localdata/lichess_db_eval.train.jsonl.zst
```

## 2) Convert JSONL.ZST to compact `.features` (recommended)

### Single output file

Use a `.features.zst` path (or `--compress-output true`) if you want compressed storage, matching `trainer/config_nnue.yaml` defaults.

Convert train:

```bash
caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.train.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.train.features.zst \
  --progress-every 1000000
```

Convert val:

```bash
caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.val.1M.features.zst \
  --progress-every 200000
```

### Sharded output (large training set)

For very large training data, write many shards under a directory (random routing per example, per-shard shuffle after ingest, random shard order each training epoch). This repo’s examples use **`--num-shards 200`** for train and **`--num-shards 1`** for validation.

Training shards (200 files under `trainer/localdata/lichess_db_eval.train.features_shards/`):

```bash
caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.train.jsonl.zst \
  --shards trainer/localdata/lichess_db_eval.train.features_shards \
  --num-shards 200 \
  --ingest-buffer-mb 1536 \
  --progress-every 1000000
```

Validation as a single shard (one file: `shard_000.features.zst` inside the directory):

```bash
caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --shards trainer/localdata/lichess_db_eval.val.1M.features_shards \
  --num-shards 1 \
  --ingest-buffer-mb 512 \
  --progress-every 200000
```

Notes:
- The converter prints running stats: `seen`, `kept`, `filtered`, `json_errors`, keep rate, speed.
- `.features` / `.features.zst` is binary; do not use `wc -l` to count samples.
- Sharded mode writes `shard_000.features.zst`, `shard_001.features.zst`, … (compressed by default). Tune `--ingest-buffer-mb` if you need a lower RAM ceiling during conversion.

## 3) Point config at `.features` data

Edit `trainer/config_nnue.yaml`.

**Single training file:**

```yaml
data:
  train_file: "trainer/localdata/lichess_db_eval.train.features.zst"
  val_file: "trainer/localdata/lichess_db_eval.val.1M.features.zst"
```

**Sharded training directory** (`train_file` is the folder containing `shard_*.features.zst`). Validation remains one file when you used `--num-shards 1`:

```yaml
data:
  train_file: "trainer/localdata/lichess_db_eval.train.features_shards"
  val_file: "trainer/localdata/lichess_db_eval.val.1M.features_shards/shard_000.features.zst"
```

## 4) Start training

```bash
caffeinate -i python3 trainer/train_nnue.py --config trainer/config_nnue.yaml
```

Optional overrides:

```bash
caffeinate -i python3 trainer/train_nnue.py \
  --config trainer/config_nnue.yaml \
  --save-dir checkpoints_nnue \
  --device auto
```

## 5) Resume from checkpoint

```bash
caffeinate -i python3 trainer/train_nnue.py \
  --config trainer/config_nnue.yaml \
  --resume checkpoints_nnue/best_model.pth
```

## End-to-end examples

### Single `.features.zst` files

```bash
caffeinate -i python3 trainer/split_data.py \
  --input trainer/localdata/lichess_db_eval.jsonl.zst \
  --val-size 1000000 \
  --val-output trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --train-output trainer/localdata/lichess_db_eval.train.jsonl.zst

caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.train.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.train.features.zst \
  --progress-every 1000000

caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.val.1M.features.zst \
  --progress-every 200000

caffeinate -i python3 trainer/train_nnue.py --config trainer/config_nnue.yaml
```

### Sharded training directory

```bash
caffeinate -i python3 trainer/split_data.py \
  --input trainer/localdata/lichess_db_eval.jsonl.zst \
  --val-size 1000000 \
  --val-output trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --train-output trainer/localdata/lichess_db_eval.train.jsonl.zst

caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.train.jsonl.zst \
  --shards trainer/localdata/lichess_db_eval.train.features_shards \
  --num-shards 200 \
  --ingest-buffer-mb 1536 \
  --progress-every 1000000

caffeinate -i python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --shards trainer/localdata/lichess_db_eval.val.1M.features_shards \
  --num-shards 1 \
  --ingest-buffer-mb 512 \
  --progress-every 200000
```

Then set `data.train_file` to `trainer/localdata/lichess_db_eval.train.features_shards` and `data.val_file` to `trainer/localdata/lichess_db_eval.val.1M.features_shards/shard_000.features.zst` in `trainer/config_nnue.yaml`, and run:

```bash
caffeinate -i python3 trainer/train_nnue.py --config trainer/config_nnue.yaml
```
