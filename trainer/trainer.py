"""
Chess Neural Network Trainer

This module provides the main training functionality for the chess evaluation neural network.
It includes training loops, validation, model saving, and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from .transformer_model import ChessTransformer, create_model, count_parameters
from .data_loader import ChessDataLoader


class ChessTrainer:
    """
    Main trainer class for chess neural network evaluation.
    
    Handles training, validation, model saving/loading, and training metrics.
    """
    
    def __init__(self, 
                 model: ChessTransformer,
                 device: str = 'auto',
                 save_dir: str = 'checkpoints',
                 cp_to_prob_scale: float = 400.0):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            save_dir: Directory to save model checkpoints
            cp_to_prob_scale: Scale factor for centipawn to probability conversion
        """
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        self.cp_to_prob_scale = cp_to_prob_scale
        
        self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.BCELoss()  # Binary cross-entropy for win probability
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the training device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        print(f"Using device: {device}")
        return torch.device(device)
    
    def setup_optimizer(self, 
                       optimizer_type: str = 'adamw',
                       learning_rate: float = 1e-4,
                       weight_decay: float = 1e-5,
                       betas: Tuple[float, float] = (0.9, 0.999)):
        """
        Setup the optimizer and learning rate scheduler.
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            betas: Beta parameters for Adam optimizer
        """
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Setup learning rate scheduler for large dataset
        # CosineAnnealingLR is better for single epoch with many batches
        # T_max will be set when we know the actual batch count
        self.scheduler = None  # Will be initialized in train_epoch
        
        print(f"Setup {optimizer_type} optimizer with lr={learning_rate}")
    
    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader = None, 
                   checkpoint_every_batches: int = None, print_every_batches: int = 100) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader for batch checkpoints
            checkpoint_every_batches: Run validation and save checkpoint every N batches
            print_every_batches: Print training progress every N batches
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        global_batch_count = 0  # Track total batches across all epochs
        
        # Initialize scheduler with actual batch count for large dataset
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(train_loader),  # Total number of batches in this epoch
                eta_min=1e-6  # Minimum learning rate
            )
            print(f"Initialized CosineAnnealingLR scheduler with T_max={len(train_loader)}")
        
        # Track loss for progress printing
        print_loss_sum = 0.0
        print_batch_count = 0
        
        if len(train_loader) == 0:
            print("Warning: No training batches available")
            return 0.0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Move data to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update learning rate (step after each batch for large dataset)
            self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            global_batch_count += 1
            
            # Accumulate loss for progress printing
            print_loss_sum += loss.item()
            print_batch_count += 1
            
            # Print progress
            if batch_idx % print_every_batches == 0:
                avg_loss_since_last_print = print_loss_sum / print_batch_count if print_batch_count > 0 else 0.0
                print(f'Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {avg_loss_since_last_print:.6f}')
                # Reset counters for next interval
                print_loss_sum = 0.0
                print_batch_count = 0
            
            # Batch-level checkpointing
            if (checkpoint_every_batches and 
                global_batch_count % checkpoint_every_batches == 0 and 
                val_loader is not None):
                self._batch_checkpoint(global_batch_count, val_loader)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                # Move data to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Print detailed validation information
        self._print_validation_details(val_loader)
        
        # Run model probes for understanding
        self._run_model_probes()
        
        # Set model back to training mode
        self.model.train()
        
        return avg_loss
    
    def _cp_to_win_probability(self, cp: float) -> float:
        """Convert centipawn evaluation to win probability using logistic function."""
        import numpy as np
        # Clamp cp to [-5000, 5000] to prevent saturation
        cp_clamped = np.clip(cp, -5000, 5000)
        
        # Use logistic function: 1 / (1 + exp(-cp/400))
        win_prob = 1.0 / (1.0 + np.exp(-cp_clamped / 400.0))
        
        return float(win_prob)
    
    def _win_probability_to_cp(self, win_prob: float) -> float:
        """Convert win probability back to centipawn evaluation (inverse of logistic function)."""
        import numpy as np
        # Clamp win_prob to [0.001, 0.999] to prevent log(0) or log(inf)
        win_prob_clamped = np.clip(win_prob, 0.001, 0.999)
        
        # Use the same scale factor as the data loader (default 400)
        scale_factor = getattr(self, 'cp_to_prob_scale', 400.0)
        
        # Inverse logistic function: cp = -scale_factor * log((1 - win_prob) / win_prob)
        cp = -scale_factor * np.log((1.0 - win_prob_clamped) / win_prob_clamped)
        
        return float(cp)
    
    
    def _print_validation_details(self, val_loader: DataLoader):
        """Print detailed validation information with random positions."""
        try:
            dataset = val_loader.dataset
            random_positions = dataset.get_random_positions(10)
            
            if not random_positions:
                print("  📊 No validation positions available for detailed logging")
                return
            
            print(f"\n  📊 Validation Details (10 random positions):")
            print(f"  {'FEN':<60} {'GT Eval':<10} {'Model Eval':<10} {'Diff':<8}")
            print(f"  {'-'*60} {'-'*10} {'-'*10} {'-'*8}")
            
            for fen, ground_truth_eval, features in random_positions:
                # Get model prediction (win probability)
                features_tensor = features.unsqueeze(0).to(self.device)  # Add batch dimension
                with torch.no_grad():
                    model_win_prob = self.model(features_tensor).cpu().item()
                
                # Convert model win probability back to centipawn evaluation
                model_eval = self._win_probability_to_cp(model_win_prob)
                
                # Calculate difference between evaluations (in centipawns)
                diff = abs(model_eval - ground_truth_eval)
                
                # Truncate FEN if too long
                fen_display = fen[:57] + "..." if len(fen) > 60 else fen
                
                print(f"  {fen_display:<60} {ground_truth_eval:<10.1f} {model_eval:<10.1f} {diff:<8.1f}")
            
            print()
            
        except Exception as e:
            print(f"  ⚠️  Could not print validation details: {e}")
            print()
    
    def _run_model_probes(self):
        """Run model probes to understand what the model has learned."""
        try:
            from model_probes import ModelProbeSuite
            from data_loader import ChessDataLoader
            
            # Create probe suite
            probe_suite = ModelProbeSuite()
            
            # Create feature extractor
            from chess_game.neural_network_evaluator import NeuralNetworkEvaluator
            feature_extractor = NeuralNetworkEvaluator()
            
            # Run a subset of key probes (not all to avoid spam)
            key_probes = [
                probe for probe in probe_suite.probes 
                if any(keyword in probe.description.lower() for keyword in ['queen', 'rook', 'center', 'development'])
            ][:5]  # Limit to 5 key probes
            
            if key_probes:
                print(f"\n  🔍 Model Understanding Probes:")
                print(f"  {'Description':<25} {'FEN1 Eval':<10} {'FEN2 Eval':<10} {'Difference':<12}")
                print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
                
                for probe in key_probes:
                    result = probe.evaluate(self.model, feature_extractor, self.device, self.cp_to_prob_scale)
                    
                    if 'error' not in result:
                        eval1 = result['fen1_eval']
                        eval2 = result['fen2_eval']
                        diff = result['eval_difference']
                        print(f"  {result['description']:<25} {eval1:<10.1f} {eval2:<10.1f} {diff:<12.1f}")
                    else:
                        print(f"  {result['description']:<25} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
                
                print()
                
        except Exception as e:
            # Silently fail to avoid disrupting training
            pass
    
    def _batch_checkpoint(self, global_batch_count: int, val_loader: DataLoader):
        """
        Perform batch-level checkpointing with validation and stats.
        
        Args:
            global_batch_count: Total number of batches processed so far
            val_loader: Validation data loader
        """
        print(f"\n🔄 Batch Checkpoint at {global_batch_count} batches:")
        
        # Run validation
        val_loss = self.validate(val_loader)
        
        # Set model back to training mode after validation
        self.model.train()
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Print stats
        print(f"  📊 Validation Loss: {val_loss:.6f}")
        print(f"  📈 Learning Rate: {current_lr:.2e}")
        print(f"  🏆 Best Val Loss: {self.best_val_loss:.6f}")
        
        # Save best model if validation improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint('best_model.pth', self.epoch, val_loss)
            print(f"  ✅ New best model saved! (val_loss: {val_loss:.6f})")
        else:
            print(f"  📝 No improvement (current: {val_loss:.6f}, best: {self.best_val_loss:.6f})")
        
        # Save regular checkpoint
        checkpoint_name = f'checkpoint_batch_{global_batch_count}.pth'
        self.save_checkpoint(checkpoint_name, self.epoch, val_loss)
        print(f"  💾 Checkpoint saved: {checkpoint_name}")
        print()  # Empty line for readability
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              save_every: int = 10,
              early_stopping_patience: int = 10,
              checkpoint_every_batches: int = None,
              print_every_batches: int = 100) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_every: Save model every N epochs
            early_stopping_patience: Stop training if no improvement for N epochs
            checkpoint_every_batches: Run validation and save checkpoint every N batches
            print_every_batches: Print training progress every N batches
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {count_parameters(self.model):,} parameters")
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            start_time = time.time()
            train_loss = self.train_epoch(train_loader, val_loader, checkpoint_every_batches, print_every_batches)
            train_time = time.time() - start_time
            
            # Validation
            start_time = time.time()
            val_loss = self.validate(val_loader)
            val_time = time.time() - start_time
            
            # Get current learning rate (updated after each batch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}, '
                  f'Train Time: {train_time:.1f}s, Val Time: {val_time:.1f}s')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(f'  -> New best model saved (val_loss: {val_loss:.6f})')
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_loss)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs '
                      f'(no improvement for {early_stopping_patience} epochs)')
                break
        
        print(f'Training completed. Best validation loss: {self.best_val_loss:.6f} '
              f'at epoch {best_epoch}')
        
        return self.training_history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
            epoch: Current epoch number
            val_loss: Current validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': self.model.__dict__  # Save model configuration
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states if available
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        })
        
        print(f'Checkpoint loaded: {filepath}')
        return checkpoint
    
    def save_model(self, filename: str):
        """
        Save only the model (for inference).
        
        Args:
            filename: Name of the model file
        """
        filepath = self.save_dir / filename
        torch.save(self.model.state_dict(), filepath)
        print(f'Model saved: {filepath}')
    
    def load_model(self, filename: str):
        """
        Load only the model (for inference).
        
        Args:
            filename: Name of the model file
        """
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f'Model loaded: {filepath}')


def train_chess_model(data_file: str,
                     config: Dict[str, Any],
                     training_config: Dict[str, Any],
                     save_dir: str = 'checkpoints') -> ChessTrainer:
    """
    High-level function to train a chess evaluation model.
    
    Args:
        data_file: Path to training data file
        config: Model configuration
        training_config: Training configuration
        save_dir: Directory to save checkpoints
        
    Returns:
        Trained ChessTrainer instance
    """
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = ChessTrainer(model, save_dir=save_dir, cp_to_prob_scale=data_config.get('cp_to_prob_scale', 400.0))
    
    # Setup optimizer
    trainer.setup_optimizer(
        optimizer_type=training_config.get('optimizer', 'adamw'),
        learning_rate=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    # Create data loaders from separate files
    train_file = data_config.get('train_file')
    val_file = data_config.get('val_file')
    
    if not train_file or not val_file:
        raise ValueError("Both train_file and val_file must be specified in config")
    
    train_loader, val_loader = ChessDataLoader.create_separate_loaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=training_config.get('batch_size', 32),
        num_workers=training_config.get('num_workers', 4),
        max_positions=training_config.get('max_positions', None),
        buffer_batches=data_config.get('buffer_batches', 20),
        cp_to_prob_scale=data_config.get('cp_to_prob_scale', 400.0)
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config.get('num_epochs', 100),
        save_every=training_config.get('save_every', 10),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
        print_every_batches=training_config.get('print_every_batches', 100)
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    from .transformer_model import ChessTransformerConfig
    
    # Configuration
    model_config = ChessTransformerConfig.get_small_config()
    training_config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'val_split': 0.1,
        'max_positions': 1000  # For testing
    }
    
    # Train model
    data_file = "/Users/rajat/chess-game/trainer/localdata/lichess_db_eval.sample.jsonl"
    
    if os.path.exists(data_file):
        print("Starting training...")
        trainer = train_chess_model(
            data_file=data_file,
            config=model_config,
            training_config=training_config,
            save_dir='test_checkpoints'
        )
        print("Training completed!")
    else:
        print(f"Data file not found: {data_file}")
