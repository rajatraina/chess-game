# NNUE Training CLI Steps

This README is only the command-line workflow to train an NNUE model.

## 0) Install dependencies

```bash
pip install torch zstandard pyyaml python-chess numpy
```

## 1) Split raw eval file into train/val

If you only have one large file (example: `trainer/localdata/lichess_db_eval.jsonl.zst`), split it:

```bash
python3 trainer/split_data.py \
  --input trainer/localdata/lichess_db_eval.jsonl.zst \
  --val-size 1000000 \
  --val-output trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --train-output trainer/localdata/lichess_db_eval.train.jsonl.zst
```

## 2) Convert JSONL.ZST to compact `.features` (recommended)

Convert train:

```bash
python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.train.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.train.features \
  --progress-every 1000000
```

Convert val:

```bash
python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.val.1M.features \
  --progress-every 200000
```

Notes:
- The converter prints running stats: `seen`, `kept`, `filtered`, `json_errors`, keep rate, speed.
- `.features` is binary, so do not use `wc -l` to count samples.

## 3) Point config to `.features` files

Edit `trainer/config_nnue.yaml`:

```yaml
data:
  train_file: "trainer/localdata/lichess_db_eval.train.features"
  val_file: "trainer/localdata/lichess_db_eval.val.1M.features"
```

## 4) Start training

```bash
python3 trainer/train_nnue.py --config trainer/config_nnue.yaml
```

Optional overrides:

```bash
python3 trainer/train_nnue.py \
  --config trainer/config_nnue.yaml \
  --save-dir checkpoints_nnue \
  --device auto
```

## 5) Resume from checkpoint

```bash
python3 trainer/train_nnue.py \
  --config trainer/config_nnue.yaml \
  --resume checkpoints_nnue/best_model.pth
```

## End-to-end example

```bash
python3 trainer/split_data.py \
  --input trainer/localdata/lichess_db_eval.jsonl.zst \
  --val-size 1000000 \
  --val-output trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --train-output trainer/localdata/lichess_db_eval.train.jsonl.zst

python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.train.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.train.features \
  --progress-every 1000000

python3 trainer/convert_eval_to_features.py \
  --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
  --output trainer/localdata/lichess_db_eval.val.1M.features \
  --progress-every 200000

python3 trainer/train_nnue.py --config trainer/config_nnue.yaml
```
