# Chess NNUE Trainer

This directory contains a complete NNUE (Efficiently Updatable Neural Networks) training system for chess position evaluation. NNUE models are much simpler and more interpretable than transformer-based approaches.

## Features

- **NNUE Architecture**: Uses piece-square table features instead of complex neural networks
- **Efficient Data Loading**: Streams data from compressed `.zst` files with batch processing
- **Centipawn Evaluation**: Directly predicts centipawn values for position evaluation
- **Simple and Fast**: Much faster training and inference than transformer models
- **Interpretable**: Piece-square table features are easy to understand and debug
- **PyTorch Integration**: Full PyTorch training pipeline with validation and checkpointing

## Architecture

The NNUE model uses a simple but effective architecture:

1. **Feature Extraction**: Converts chess position to piece-square table features (768 features)
2. **Hidden Layers**: 2-3 fully connected layers with ReLU activation
3. **Output**: Single centipawn evaluation value
4. **Training**: MSE loss for centipawn prediction

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

### 2. Train with Configuration File

```bash
# Use default config file
python3 trainer/train_nnue.py

# Use custom config file
python3 trainer/train_nnue.py --config my_experiment_config.yaml

# With custom save directory
python3 trainer/train_nnue.py --config config_nnue.yaml --save-dir my_nnue_checkpoints
```

### 3. Resume Training

```bash
python3 trainer/train_nnue.py \
    --config config_nnue.yaml \
    --resume checkpoints_nnue/best_model.pth
```

## Configuration Examples

### Default Configuration
The `config_nnue.yaml` file contains a balanced configuration:
- 256 hidden size, 2 layers
- 64 batch size, 0.001 learning rate, 50 epochs
- 1M max training positions
- Uses complete validation set

## Creating Custom Configurations

For different experiments, create new config files:

```bash
# Copy the default config
cp config_nnue.yaml my_experiment.yaml

# Edit my_experiment.yaml with your changes
# Then run:
python3 trainer/train_nnue.py --config my_experiment.yaml
```

The config file contains all settings:
- Model architecture (hidden size, layers, dropout)
- Training parameters (batch size, learning rate, epochs)
- Data loading settings (file paths, workers, memory)
- Hardware configuration (device, mixed precision)

## Model Integration

The trained NNUE model can be integrated with the existing chess engine:

```python
from trainer.nnue_model import create_nnue_model
from trainer.nnue_trainer import NNUETrainer
import torch

# Load trained NNUE model
model = create_nnue_model(config['model'])
trainer = NNUETrainer(model)
trainer.load_model('checkpoints_nnue/best_model.pth')

# Use for evaluation
board = chess.Board()
features = trainer.feature_extractor._board_to_features(board)
features_tensor = torch.from_numpy(features).float().unsqueeze(0)
centipawn_eval = model(features_tensor).item()
```

## File Structure

```
trainer/
├── __init__.py              # Package initialization
├── nnue_model.py           # NNUE model architecture
├── nnue_trainer.py         # NNUE training loop and utilities
├── nnue_data_loader.py     # NNUE data loading and preprocessing
├── train_nnue.py           # Main NNUE training script
├── config_nnue.yaml        # NNUE configuration file
├── README.md               # This file
└── localdata/              # Training data directory
    ├── lichess_db_eval.train.jsonl.zst
    └── lichess_db_eval.val.jsonl.zst
```

## Training Tips

1. **Start Small**: Use the `small` preset for initial experiments
2. **Monitor Validation**: Watch for overfitting with validation loss
3. **Adjust Learning Rate**: Lower for stable training, higher for faster convergence
4. **Batch Size**: Larger batches for stability, smaller for memory constraints
5. **Data Size**: Start with limited data for faster iterations
6. **NNUE Benefits**: Much faster training and inference than transformer models

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
