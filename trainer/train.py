#!/usr/bin/env python3
"""
Main Training Script for Chess Neural Network

This script provides a command-line interface for training chess evaluation neural networks.
It supports various configuration options and presets for different training scenarios.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from .config import TrainingConfig, ConfigPresets, ConfigValidator
from .transformer_model import create_model
from .trainer import ChessTrainer
from .data_loader import ChessDataLoader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train a chess evaluation neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data-file', '-d',
        type=str,
        help='Path to training data file (.zst format)'
    )
    
    # Model configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--preset', '-p',
        choices=['small', 'medium', 'large', 'fast'],
        default='medium',
        help='Use predefined configuration preset'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--num-epochs', '-e',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--max-positions',
        type=int,
        help='Maximum number of positions to use for training'
    )
    
    # Output arguments
    parser.add_argument(
        '--save-dir', '-s',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name for the saved model'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        help='Number of worker processes for data loading'
    )
    
    # Training control
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation (requires --resume)'
    )
    
    # Other arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--buffer-batches',
        type=int,
        default=20,
        help='Number of batches to read in one go (buffer_size = batch_size × buffer_batches)'
    )
    
    parser.add_argument(
        '--checkpoint-every-batches',
        type=int,
        help='Save checkpoint and run validation every N batches'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        # Use preset
        preset_map = {
            'small': ConfigPresets.get_small_config,
            'medium': ConfigPresets.get_medium_config,
            'large': ConfigPresets.get_large_config,
            'fast': ConfigPresets.get_fast_training_config
        }
        config = preset_map[args.preset]()
    
    # Override with command line arguments
    if args.data_file:
        config.set('paths.data_file', args.data_file)
    
    if args.save_dir:
        config.set('paths.save_dir', args.save_dir)
    
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    
    if args.num_epochs:
        config.set('training.num_epochs', args.num_epochs)
    
    if args.max_positions:
        config.set('data.max_positions', args.max_positions)
    
    if args.device:
        config.set('hardware.device', args.device)
    
    if args.num_workers:
        config.set('data.num_workers', args.num_workers)
    
    if args.checkpoint_every_batches:
        config.set('training.checkpoint_every_batches', args.checkpoint_every_batches)
    
    # Validate configuration
    print("Validating configuration...")
    issues = ConfigValidator.validate(config)
    ConfigValidator.print_validation_results(issues)
    
    if issues['errors']:
        print("❌ Configuration has errors. Exiting.")
        sys.exit(1)
    
    # Check data file exists
    data_file = config.get('paths.data_file')
    if not data_file:
        print("❌ No data file specified. Use --data-file or set 'paths.data_file' in config.")
        sys.exit(1)
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    # Create save directory
    save_dir = config.get('paths.save_dir')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.yaml')
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Create model
    print("Creating model...")
    model_config = config.get('model')
    model = create_model(model_config)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = ChessTrainer(
        model=model,
        device=config.get('hardware.device'),
        save_dir=save_dir
    )
    
    # Setup optimizer
    training_config = config.get('training')
    trainer.setup_optimizer(
        optimizer_type=training_config.get('optimizer'),
        learning_rate=training_config.get('learning_rate'),
        weight_decay=training_config.get('weight_decay'),
        betas=tuple(training_config.get('betas'))
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Create data loaders
    print("Creating data loaders...")
    data_config = config.get('data')
    
    # For streaming data loader, we must use num_workers=0
    if data_config.get('num_workers', 0) > 0:
        print("Warning: Setting num_workers to 0 for streaming data loader")
        data_config['num_workers'] = 0
    
    buffer_batches = args.buffer_batches or data_config.get('buffer_batches', 20)
    train_file = data_config.get('train_file')
    val_file = data_config.get('val_file')
    
    if not train_file or not val_file:
        raise ValueError("Both train_file and val_file must be specified in config")
    
    train_loader, val_loader = ChessDataLoader.create_separate_loaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=training_config.get('batch_size'),
        num_workers=0,  # Must be 0 for streaming
        max_positions=data_config.get('max_positions'),
        buffer_batches=buffer_batches,
        cp_to_prob_scale=data_config.get('cp_to_prob_scale', 400.0)
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Run validation only if requested
    if args.validate_only:
        if not args.resume:
            print("❌ --validate-only requires --resume")
            sys.exit(1)
        
        print("Running validation...")
        val_loss = trainer.validate(val_loader)
        print(f"Validation loss: {val_loss:.6f}")
        return
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config.get('num_epochs'),
        save_every=training_config.get('save_every'),
        early_stopping_patience=training_config.get('early_stopping_patience'),
        checkpoint_every_batches=training_config.get('checkpoint_every_batches'),
        print_every_batches=training_config.get('print_every_batches', 100)
    )
    
    # Save final model
    model_name = args.model_name or f"chess_model_{args.preset}"
    trainer.save_model(f"{model_name}.pth")
    
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Model saved as: {model_name}.pth")


if __name__ == "__main__":
    main()
