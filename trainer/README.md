# Chess Neural Network Trainer

This directory contains a complete neural network training system for chess position evaluation using transformer architectures.

## Features

- **Transformer-based Architecture**: Uses attention mechanisms instead of convolutional layers
- **Efficient Data Loading**: Streams data from compressed `.zst` files with batch processing (compressed files only)
- **Side-to-move Perspective**: Always evaluates from the perspective of the side to move
- **Win Probability Output**: Converts centipawn evaluations to win probabilities using logistic function
- **Flexible Configuration**: Supports multiple model sizes and training configurations
- **PyTorch Integration**: Full PyTorch training pipeline with validation and checkpointing

## Architecture

The neural network uses a transformer architecture specifically designed for chess:

1. **Input Projection**: Converts 8×8×18 board representation to transformer input
2. **Positional Encoding**: Adds position information for the 64 squares
3. **Transformer Encoder**: Multi-head attention layers for position understanding
4. **Output Head**: Produces win probability [0, 1] from side-to-move perspective

## Data Format

The trainer expects training data in the following JSON format (one position per line):

```json
{
  "fen": "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -",
  "evals": [
    {
      "pvs": [
        {
          "cp": 48,
          "line": "f7g7 e6e2 h8d8 e2d2 b7b5 c4b3 a6a5 a2a3 g7f6 b3a2"
        }
      ],
      "knodes": 644403,
      "depth": 55
    }
  ]
}
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch zstandard
```

### 2. Train with Preset Configuration

```bash
# Small model for testing
python -m trainer.train --data-file localdata/lichess_db_eval.jsonl.zst --preset small

# Medium model for balanced performance
python -m trainer.train --data-file localdata/lichess_db_eval.jsonl.zst --preset medium

# Large model for maximum performance
python -m trainer.train --data-file localdata/lichess_db_eval.jsonl.zst --preset large
```

### 3. Custom Training

```bash
python -m trainer.train \
    --data-file localdata/lichess_db_eval.jsonl.zst \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --num-epochs 50 \
    --max-positions 100000 \
    --save-dir my_checkpoints
```

### 4. Resume Training

```bash
python -m trainer.train \
    --data-file localdata/lichess_db_eval.jsonl.zst \
    --resume checkpoints/best_model.pth \
    --preset medium
```

## Configuration Presets

### Small (Testing)
- 128 dimensions, 4 heads, 3 layers
- Fast training, good for development
- ~1M parameters

### Medium (Balanced)
- 256 dimensions, 8 heads, 6 layers
- Good balance of performance and speed
- ~5M parameters

### Large (Performance)
- 512 dimensions, 16 heads, 12 layers
- Maximum performance, slower training
- ~50M parameters

### Fast (Quick Iterations)
- 64 dimensions, 2 heads, 2 layers
- Very fast training for quick experiments
- ~100K parameters

## Custom Configuration

Create a JSON configuration file:

```json
{
  "model": {
    "d_model": 256,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 1024,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "val_split": 0.1,
    "optimizer": "adamw"
  },
  "data": {
    "num_workers": 4,
    "max_positions": null
  }
}
```

Then use it:

```bash
python -m trainer.train --data-file data.jsonl.zst --config my_config.yaml
```

## Model Integration

The trained model can be integrated with the existing chess engine:

```python
from trainer.transformer_model import create_model
from trainer.trainer import ChessTrainer
import torch

# Load trained model
model = create_model(config)
trainer = ChessTrainer(model)
trainer.load_model('checkpoints/best_model.pth')

# Use for evaluation
board = chess.Board()
features = trainer.feature_extractor._board_to_features(board)
features_tensor = torch.from_numpy(features).float().unsqueeze(0)
win_prob = model(features_tensor).item()
```

## File Structure

```
trainer/
├── __init__.py              # Package initialization
├── transformer_model.py     # Transformer architecture
├── data_loader.py          # Data loading and preprocessing
├── trainer.py              # Training loop and utilities
├── config.py               # Configuration management
├── train.py                # Main training script
├── README.md               # This file
└── localdata/              # Training data directory
    ├── lichess_db_eval.jsonl.zst
    └── lichess_db_eval.sample.jsonl
```

## Training Tips

1. **Start Small**: Use the `small` or `fast` preset for initial experiments
2. **Monitor Validation**: Watch for overfitting with validation loss
3. **Adjust Learning Rate**: Lower for stable training, higher for faster convergence
4. **Batch Size**: Larger batches for stability, smaller for memory constraints
5. **Data Size**: Start with limited data (`--max-positions`) for faster iterations

## Hardware Requirements

- **CPU**: Multi-core recommended for data loading
- **RAM**: 8GB+ recommended for large datasets
- **GPU**: CUDA/MPS support for faster training (optional)
- **Storage**: SSD recommended for data loading performance

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller model
2. **Slow Training**: Increase `num_workers` or use GPU
3. **Poor Convergence**: Lower learning rate or check data quality
4. **Data Loading Errors**: Verify `.zst` file format and permissions

### Performance Optimization

1. Use GPU if available (`--device cuda`)
2. Increase `num_workers` for faster data loading
3. Use `pin_memory=True` for GPU training
4. Consider mixed precision training for large models
