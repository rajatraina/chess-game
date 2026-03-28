#!/usr/bin/env python3
"""
NNUE Training Script for Chess Position Evaluation

Trains an NNUE model on White-perspective centipawn regression targets (mate rows
skipped). Loss: SmoothL1, MSE, or win_prob_bce (soft BCE on sigma(cp/s)); labels are
clamped and optionally scaled.

Usage:
    python3 trainer/train_nnue.py --config trainer/config_nnue.yaml
    python3 trainer/train_nnue.py --config my_experiment.yaml

Checkpoints from the old sigmoid+BCE objective are incompatible with this head.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import chess
import numpy as np

from trainer.nnue_model import create_nnue_model, count_parameters, NNUEFeatureExtractor
from trainer.nnue_trainer import NNUETrainer
from trainer.nnue_data_loader import NNUEDataLoader, apply_cp_regression_target


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def _run_preflight_checks(
    train_file: str,
    cp_label_clip_min: float,
    cp_label_clip_max: float,
    cp_target_scale: float,
) -> None:
    """Fail fast on common semantic mismatches before a long run."""
    print("\nRunning NNUE preflight checks...")

    samples = NNUEDataLoader.inspect_targets(
        train_file,
        max_samples=3,
        cp_label_clip_min=cp_label_clip_min,
        cp_label_clip_max=cp_label_clip_max,
        cp_target_scale=cp_target_scale,
    )
    if not samples:
        raise RuntimeError(
            "Could not parse any labeled training targets (cp/mate-mapped)."
        )

    print("Sample parsed targets (cp + mate-mapped labels):")
    for idx, sample in enumerate(samples, start=1):
        side = sample.get("side_to_move", "unknown")
        depth = sample.get("selected_depth")
        raw_cp = sample.get("raw_cp")
        cp_clamped = sample.get("cp_clamped")
        reg_target = float(sample.get("regression_target", 0.0))
        if cp_clamped is None:
            print(
                f"  [{idx}] stm={side}, depth={depth}, raw_cp={raw_cp}, "
                f"regression_target={reg_target:.4f}"
            )
        else:
            print(
                f"  [{idx}] stm={side}, depth={depth}, raw_cp={raw_cp}, "
                f"clamped_cp={float(cp_clamped):.1f}, regression_target={reg_target:.4f}"
            )

    if cp_label_clip_min >= cp_label_clip_max:
        raise RuntimeError("cp_label_clip_min must be < cp_label_clip_max")

    z = apply_cp_regression_target(0.0, cp_label_clip_min, cp_label_clip_max, cp_target_scale)
    if abs(z) > 1e-6:
        raise RuntimeError("Regression target for 0 cp should be 0 after clamp/scale.")

    extractor = NNUEFeatureExtractor()
    start_white = chess.Board()
    start_black = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    white_features = extractor.board_to_features(start_white)
    black_features = extractor.board_to_features(start_black)
    if not np.array_equal(white_features[:-1], black_features[:-1]):
        raise RuntimeError("Feature orientation changed outside the side-to-move feature.")
    if white_features[-1] != 1.0 or black_features[-1] != 0.0:
        raise RuntimeError("Side-to-move feature is encoded incorrectly.")

    white_up_queen = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    equal_kings = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    black_up_queen = chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1")
    white_feats = extractor.board_to_features(white_up_queen)
    equal_feats = extractor.board_to_features(equal_kings)
    black_feats = extractor.board_to_features(black_up_queen)
    if not (white_feats[776] > equal_feats[776] and black_feats[781] > equal_feats[781]):
        raise RuntimeError("Piece-count features do not reflect material imbalances correctly.")

    print("Preflight checks passed.")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train NNUE model for chess position evaluation')
    parser.add_argument('--config', type=str, default='config_nnue.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    training_config = config['training']
    data_config = config['data']
    hardware_config = config.get('hardware', {})
    path_config = config.get('paths', {})
    nnue_cfg = config.get('nnue', {})

    train_file = data_config['train_file']
    val_file = data_config['val_file']
    cp_label_clip_min = float(nnue_cfg.get('cp_label_clip_min', -2000.0))
    cp_label_clip_max = float(nnue_cfg.get('cp_label_clip_max', 2000.0))
    cp_target_scale = float(nnue_cfg.get('cp_target_scale', 1.0))

    save_dir = args.save_dir or path_config.get('save_dir', 'checkpoints_nnue')
    device = args.device or hardware_config.get('device', 'auto')

    if not os.path.exists(train_file):
        print(f"Error: Training path {train_file} not found (file or directory of .features shards)")
        sys.exit(1)
    if not os.path.exists(val_file):
        print(f"Error: Validation file {val_file} not found")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("NNUE TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Model: {config['model']}")
    print(f"Training: {training_config}")
    print(f"Data: {data_config}")
    print(f"NNUE: {nnue_cfg}")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
    print("=" * 50)

    model = create_nnue_model(config['model'])
    print(f"\nCreated NNUE model with {count_parameters(model):,} parameters")

    trainer = NNUETrainer(
        model,
        device=device,
        save_dir=save_dir,
        regression_loss=str(nnue_cfg.get('regression_loss', 'smooth_l1')),
        huber_beta=float(nnue_cfg.get('huber_beta', 100.0)),
        cp_target_scale=cp_target_scale,
        win_prob_scale=float(nnue_cfg.get('win_prob_scale', 400.0)),
        cp_regularization_weight=float(nnue_cfg.get('cp_regularization_weight', 0.0)),
        cp_regularization_loss=str(nnue_cfg.get('cp_regularization_loss', 'smooth_l1')),
        cp_regularization_beta=float(nnue_cfg.get('cp_regularization_beta', 100.0)),
        balanced_eval_threshold_cp=float(nnue_cfg.get('balanced_eval_threshold_cp', 100.0)),
    )

    trainer.setup_optimizer(
        optimizer_type=training_config.get('optimizer', 'adamw'),
        learning_rate=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 1e-4),
        betas=tuple(training_config.get('betas', [0.9, 0.999])),
    )
    trainer.gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)

    scheduler_type = training_config.get('scheduler', 'step')
    scheduler_kwargs = {
        'step_size': training_config.get('step_size', 15),
        'step_size_batches': training_config.get('step_size_batches', None),
        'gamma': training_config.get('gamma', 0.5),
        'factor': training_config.get('factor', 0.5),
        'patience': training_config.get('patience', 5),
        'min_lr': training_config.get('min_lr', 1e-6),
        'T_max': training_config.get('T_max', 50),
        'eta_min': training_config.get('eta_min', 0),
    }
    trainer.setup_scheduler(scheduler_type=scheduler_type, **scheduler_kwargs)

    if args.resume:
        print(f"Resuming training from: {args.resume}")
        trainer.load_model(args.resume)

    _run_preflight_checks(
        train_file,
        cp_label_clip_min=cp_label_clip_min,
        cp_label_clip_max=cp_label_clip_max,
        cp_target_scale=cp_target_scale,
    )

    print("Creating data loaders...")
    train_loader, val_loader = NNUEDataLoader.create_data_loaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=training_config.get('batch_size', 64),
        max_train_positions=data_config.get('max_train_positions'),
        max_val_positions=data_config.get('max_val_positions'),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        cp_label_clip_min=cp_label_clip_min,
        cp_label_clip_max=cp_label_clip_max,
        cp_target_scale=cp_target_scale,
    )

    try:
        print(f"Training batches: {len(train_loader)}")
    except TypeError:
        print("Training batches: Streaming (unknown count)")
    print(f"Validation batches: {len(val_loader)}")

    print("\nTesting data loading...")
    for i, (features, targets) in enumerate(train_loader):
        print(f"Batch {i}: features shape {features.shape}, targets shape {targets.shape}")
        print(f"Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
        if i >= 2:
            break

    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config.get('num_epochs', 50),
        save_every=training_config.get('save_every', 5),
        save_every_batches=training_config.get('save_every_batches', 100000),
        validate_every_batches=training_config.get('validate_every_batches', None),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
        print_every=1,
        probe_every_batches=training_config.get('probe_every_batches'),
        show_learning_curve=training_config.get('show_learning_curve', False),
    )

    trainer.save_model('final_model.pth')

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
