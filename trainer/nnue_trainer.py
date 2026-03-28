"""
NNUE Trainer for Chess Position Evaluation

Trains an NNUE model with White-perspective centipawn regression targets.
Loss options: SmoothL1, MSE, or soft-label BCE on win probability derived from
native cp predictions (see cp_winprob_utils). The engine-facing evaluator maps
network output back to centipawns (multiplies by cp_target_scale when training
used normalized targets).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from .nnue_model import NNUE, create_nnue_model, count_parameters
from .nnue_data_loader import NNUEDataLoader
from .cp_winprob_utils import cp_to_win_prob_tensor, soft_binary_cross_entropy_probs


class NNUETrainer:
    """
    Trainer class for NNUE chess evaluation models.
    
    Handles training, validation, checkpointing, resume state, and probe logging.
    Uses regression loss (SmoothL1, MSE, or win_prob_bce) on clipped/scaled cp targets.
    Optionally adds a cp-space regularizer on top of win_prob_bce to improve absolute
    centipawn calibration on balanced positions.
    """
    
    def __init__(self, 
                 model: NNUE,
                 device: str = 'auto',
                 save_dir: str = 'checkpoints_nnue',
                 regression_loss: str = 'smooth_l1',
                 huber_beta: float = 100.0,
                 cp_target_scale: float = 1.0,
                 win_prob_scale: float = 400.0,
                 cp_regularization_weight: float = 0.0,
                 cp_regularization_loss: str = 'smooth_l1',
                 cp_regularization_beta: float = 100.0,
                 balanced_eval_threshold_cp: float = 100.0):
        """
        Initialize the NNUE trainer.
        
        Args:
            model: The NNUE model to train
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            save_dir: Directory to save model checkpoints
            regression_loss: 'smooth_l1' (Huber), 'mse', or 'win_prob_bce'
            huber_beta: beta for nn.SmoothL1Loss (centipawn units of target tensor)
            cp_target_scale: multiply probe outputs / match inference (same as nnue.cp_target_scale)
            win_prob_scale: s in sigma(cp/s) for win_prob_bce (centipawns; default 400)
            cp_regularization_weight: lambda on cp-space regularizer when using win_prob_bce
            cp_regularization_loss: cp-space penalty ('smooth_l1', 'l1', or 'mse')
            cp_regularization_beta: beta for cp-space SmoothL1 in centipawn units
            balanced_eval_threshold_cp: validation slice threshold for near-equal positions
        """
        self.model = model
        self.cp_target_scale = float(cp_target_scale) if cp_target_scale else 1.0
        if self.cp_target_scale <= 0:
            self.cp_target_scale = 1.0
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'val_win_prob_bce': [],
            'val_cp_regularizer': [],
            'val_weighted_cp_regularizer': [],
            'val_cp_mae': [],
            'val_balanced_cp_mae': [],
        }
        self.run_state = {
            'total_batches': 0,
            'last_saved_batch': 0,
            'last_lr_step_batch': 0,
            'last_validated_batch': 0,
            'epochs_without_improvement': 0,
            'batches_in_epoch': 0,
        }
        self.gradient_clip_norm = 1.0
        self.resume_skip_batches = 0
        self.show_learning_curve = False
        self._curve_train: List[float] = []
        self._curve_val: List[float] = []
        self._curve_x_examples: List[float] = []
        self._lc_batch_size = 1
        self._lc_epoch_end_examples: List[int] = []
        self._learning_curve_fig = None
        self._learning_curve_ax = None
        self._learning_curve_ax2 = None
        self._learning_curve_train_line = None
        self._learning_curve_val_line = None
        self._lc_import_warned = False
        self._last_val_cp_mae: Optional[float] = None
        self._last_val_balanced_cp_mae: Optional[float] = None
        self._last_val_metrics: Dict[str, Optional[float]] = {}
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        rl = (regression_loss or "smooth_l1").lower().replace("-", "_")
        if rl not in ("mse", "smooth_l1", "win_prob_bce"):
            raise ValueError(
                f"Unknown regression_loss: {regression_loss!r}. "
                "Use 'smooth_l1', 'mse', or 'win_prob_bce'."
            )
        self.regression_loss_name = rl
        self.win_prob_scale = float(win_prob_scale) if win_prob_scale and win_prob_scale > 0 else 400.0
        self.cp_regularization_weight = max(0.0, float(cp_regularization_weight or 0.0))
        cpr = (cp_regularization_loss or "smooth_l1").lower().replace("-", "_")
        if cpr not in ("smooth_l1", "l1", "mse"):
            raise ValueError(
                f"Unknown cp_regularization_loss: {cp_regularization_loss!r}. "
                "Use 'smooth_l1', 'l1', or 'mse'."
            )
        self.cp_regularization_loss_name = cpr
        self.cp_regularization_beta = (
            float(cp_regularization_beta) if cp_regularization_beta and cp_regularization_beta > 0 else 100.0
        )
        self.balanced_eval_threshold_cp = max(0.0, float(balanced_eval_threshold_cp or 0.0))
        if rl == "win_prob_bce":
            self.criterion = None
        elif rl == "mse":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.SmoothL1Loss(beta=float(huber_beta))
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the training device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                print(f"Warning: CUDA requested but not available. Available options:")
                if torch.backends.mps.is_available():
                    print(f"  - MPS (Apple Silicon GPU) is available")
                print(f"  - Falling back to CPU")
                print(f"  - Use --device auto to automatically select the best device")
                device = 'cpu'
        elif device == 'mps':
            if not torch.backends.mps.is_available():
                print(f"Warning: MPS requested but not available. Falling back to CPU.")
                device = 'cpu'
        
        print(f"Using device: {device}")
        return torch.device(device)

    def _cp_space_tensors(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return targets and predictions in centipawn units."""
        return targets * self.cp_target_scale, predictions * self.cp_target_scale

    def _cp_regularization_tensor(self, yhat_cp: torch.Tensor, y_cp: torch.Tensor) -> torch.Tensor:
        """Batch mean cp-space penalty used to stabilize absolute calibration."""
        if self.cp_regularization_loss_name == "smooth_l1":
            return F.smooth_l1_loss(yhat_cp, y_cp, beta=self.cp_regularization_beta)
        if self.cp_regularization_loss_name == "l1":
            return F.l1_loss(yhat_cp, y_cp)
        return F.mse_loss(yhat_cp, y_cp)

    def _compute_loss_components(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute the total loss and its auditable components."""
        y_cp, yhat_cp = self._cp_space_tensors(predictions, targets)
        zero = predictions.new_zeros(())

        if self.regression_loss_name == "win_prob_bce":
            p_star = cp_to_win_prob_tensor(y_cp, self.win_prob_scale)
            p_hat = cp_to_win_prob_tensor(yhat_cp, self.win_prob_scale)
            win_prob_bce = soft_binary_cross_entropy_probs(p_hat, p_star)
            cp_regularizer = zero
            weighted_cp_regularizer = zero
            total = win_prob_bce
            if self.cp_regularization_weight > 0:
                cp_regularizer = self._cp_regularization_tensor(yhat_cp, y_cp)
                weighted_cp_regularizer = cp_regularizer * self.cp_regularization_weight
                total = total + weighted_cp_regularizer
            return {
                'total': total,
                'win_prob_bce': win_prob_bce,
                'cp_regularizer': cp_regularizer,
                'weighted_cp_regularizer': weighted_cp_regularizer,
                'y_cp': y_cp,
                'yhat_cp': yhat_cp,
            }

        base = self.criterion(predictions, targets)
        return {
            'total': base,
            'win_prob_bce': zero,
            'cp_regularizer': zero,
            'weighted_cp_regularizer': zero,
            'y_cp': y_cp,
            'yhat_cp': yhat_cp,
        }

    def _regression_loss_tensor(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Batch mean loss for the configured regression objective."""
        return self._compute_loss_components(predictions, targets)['total']

    def _format_cp_regularizer_name(self) -> str:
        if self.cp_regularization_loss_name == "smooth_l1":
            return f"smooth_l1(cp, beta={self.cp_regularization_beta:g})"
        return f"{self.cp_regularization_loss_name}(cp)"

    def _format_last_val_metrics(self) -> str:
        parts: List[str] = []
        metrics = self._last_val_metrics or {}
        if self.regression_loss_name == "win_prob_bce":
            bce = metrics.get('win_prob_bce')
            if bce is not None:
                parts.append(f"Val BCE: {bce:.4f}")
            if self.cp_regularization_weight > 0:
                cp_reg = metrics.get('cp_regularizer')
                weighted_cp_reg = metrics.get('weighted_cp_regularizer')
                if cp_reg is not None:
                    parts.append(f"Val CP Reg: {cp_reg:.2f}")
                if weighted_cp_reg is not None:
                    parts.append(f"Weighted: {weighted_cp_reg:.4f}")
        if self._last_val_cp_mae is not None:
            parts.append(f"Val MAE cp: {self._last_val_cp_mae:.2f}")
        if self._last_val_balanced_cp_mae is not None:
            parts.append(
                f"Val MAE |cp|<={self.balanced_eval_threshold_cp:g}: {self._last_val_balanced_cp_mae:.2f}"
            )
        return (", " + ", ".join(parts)) if parts else ""
    
    def setup_optimizer(self, 
                       optimizer_type: str = 'adamw',
                       learning_rate: float = 1e-3,
                       weight_decay: float = 1e-4,
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
                betas=betas,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        print(f"Optimizer: {optimizer_type}, LR: {learning_rate}, Weight decay: {weight_decay}")
    
    def setup_scheduler(self, scheduler_type: str = 'step', **kwargs):
        """
        Setup the learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')
            **kwargs: Additional arguments for the scheduler
                - step_size: For 'step' scheduler, can be int (epochs) or dict with 'epochs'/'batches' keys
                - step_size_batches: For batch-based step scheduler (alternative to step_size dict)
        """
        if self.optimizer is None:
            raise ValueError("Must setup optimizer before scheduler")
        
        # Track if scheduler should step on batches vs epochs
        self.scheduler_step_on_batches = False
        self.scheduler_step_size_batches = None
        
        if scheduler_type == 'step':
            step_size = kwargs.get('step_size', 10)
            step_size_batches = kwargs.get('step_size_batches', None)
            
            # Support batch-based stepping
            if step_size_batches is not None:
                # Batch-based: step every N batches
                self.scheduler_step_on_batches = True
                self.scheduler_step_size_batches = step_size_batches
                # Use a large step_size for StepLR (won't be used, we'll step manually)
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=999999,  # Large value so it never triggers automatically
                    gamma=kwargs.get('gamma', 0.5)
                )
                print(f"Scheduler: {scheduler_type} (batch-based, step every {step_size_batches:,} batches)")
            else:
                # Epoch-based: step every N epochs
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=step_size,
                    gamma=kwargs.get('gamma', 0.5)
                )
                print(f"Scheduler: {scheduler_type} (epoch-based, step every {step_size} epochs)")
        elif scheduler_type == 'cosine':
            # Cosine annealing - smooth decay over T_max steps
            # T_max can be epochs or batches depending on when scheduler.step() is called
            T_max = kwargs.get('T_max', 50)
            eta_min = kwargs.get('eta_min', 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
            print(f"Scheduler: {scheduler_type} (T_max={T_max}, eta_min={eta_min})")
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau - adaptive, reduces when validation loss plateaus
            # Most robust: no manual step sizes needed, adapts to training progress
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('min_lr', 1e-6),
                verbose=True  # Print when LR is reduced
            )
            print(f"Scheduler: {scheduler_type} (factor={kwargs.get('factor', 0.5)}, patience={kwargs.get('patience', 5)})")
        elif scheduler_type == 'exponential':
            # Exponential decay - continuous decay every step/epoch
            # No manual step size needed, just decay factor
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
            print(f"Scheduler: {scheduler_type} (gamma={gamma})")
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _record_lc_epoch_boundary(self, total_batches: int) -> None:
        """Record cumulative training examples at end of an epoch (for top-axis epoch ticks)."""
        if not self.show_learning_curve:
            return
        ex = int(total_batches * self._lc_batch_size)
        if not self._lc_epoch_end_examples or self._lc_epoch_end_examples[-1] != ex:
            self._lc_epoch_end_examples.append(ex)

    def _format_lc_examples_axis(self, x: float, _pos: Optional[int]) -> str:
        if x >= 1e6:
            return f"{x / 1e6:.2f}M"
        if x >= 1e3:
            return f"{x / 1e3:.1f}k"
        return f"{int(round(x)):,}"

    def _update_learning_curve_epoch_axis(self) -> None:
        if self._learning_curve_ax2 is None:
            return
        self._learning_curve_ax2.set_xlim(self._learning_curve_ax.get_xlim())
        ends = self._lc_epoch_end_examples
        positions = [0.0] + [float(e) for e in ends]
        labels = [str(i) for i in range(len(positions))]
        self._learning_curve_ax2.set_xticks(positions, labels=labels)
        self._learning_curve_ax2.tick_params(top=True, labeltop=True)

    def _append_learning_curve_point(
        self, train_loss: float, val_loss: float, *, total_batches: int
    ) -> None:
        """Record (train, val) loss vs cumulative examples; refresh live plot if enabled."""
        if not self.show_learning_curve:
            return
        x_ex = float(total_batches * self._lc_batch_size)
        self._curve_train.append(float(train_loss))
        self._curve_val.append(float(val_loss))
        self._curve_x_examples.append(x_ex)
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            if not self._lc_import_warned:
                print("Warning: matplotlib is not installed; cannot show learning curve (pip install matplotlib).")
                self._lc_import_warned = True
            self.show_learning_curve = False
            return

        xs = self._curve_x_examples
        if self._learning_curve_fig is None:
            plt.ion()
            self._learning_curve_fig, self._learning_curve_ax = plt.subplots(figsize=(8, 5))
            self._learning_curve_ax.set_xlabel("Training examples seen (cumulative)")
            self._learning_curve_ax.xaxis.set_major_formatter(
                FuncFormatter(self._format_lc_examples_axis)
            )
            self._learning_curve_ax.set_ylabel("Loss")
            self._learning_curve_ax.set_title("NNUE train vs validation loss")
            (self._learning_curve_train_line,) = self._learning_curve_ax.plot(
                xs, self._curve_train, "b.-", label="Train", markersize=4
            )
            (self._learning_curve_val_line,) = self._learning_curve_ax.plot(
                xs, self._curve_val, "r.-", label="Val", markersize=4
            )
            self._learning_curve_ax.legend(loc="upper right")
            self._learning_curve_ax.grid(True, alpha=0.3)
            self._learning_curve_ax2 = self._learning_curve_ax.twiny()
            self._learning_curve_ax2.set_xlabel("Epoch (tick at epoch start; 0 = start of training)")
            self._learning_curve_ax2.xaxis.set_label_position("top")
            self._update_learning_curve_epoch_axis()
            self._learning_curve_fig.tight_layout()
            try:
                self._learning_curve_fig.show()
            except Exception:
                pass
        else:
            self._learning_curve_train_line.set_data(xs, self._curve_train)
            self._learning_curve_val_line.set_data(xs, self._curve_val)
            self._learning_curve_ax.relim()
            self._learning_curve_ax.autoscale_view()
            self._update_learning_curve_epoch_axis()
        self._learning_curve_fig.canvas.draw()
        self._learning_curve_fig.canvas.flush_events()
        plt.pause(0.001)

    def _snapshot_run_state(self,
                            total_batches: int,
                            last_saved_batch: int,
                            last_lr_step_batch: int,
                            last_validated_batch: int,
                            epochs_without_improvement: int,
                            batches_in_epoch: int) -> None:
        """Persist counters that control long-running training cadence."""
        self.run_state.update({
            'total_batches': total_batches,
            'last_saved_batch': last_saved_batch,
            'last_lr_step_batch': last_lr_step_batch,
            'last_validated_batch': last_validated_batch,
            'epochs_without_improvement': epochs_without_improvement,
            'batches_in_epoch': batches_in_epoch,
        })

    def train_epoch(self,
                    train_loader: DataLoader,
                    probe_every_batches: Optional[int] = None,
                    val_loader: Optional[DataLoader] = None,
                    total_batches: int = 0,
                    last_saved_batch: int = 0,
                    last_lr_step_batch: int = 0,
                    last_validated_batch: int = 0,
                    validate_every_batches: Optional[int] = None,
                    save_every_batches: Optional[int] = None,
                    scheduler: Optional[Any] = None,
                    scheduler_step_size_batches: Optional[int] = None,
                    current_epoch: int = 0,
                    epochs_without_improvement: int = 0,
                    skip_batches: int = 0) -> Tuple[float, int, int, int, int, int]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            probe_every_batches: Run model probes every N batches
            val_loader: Validation loader for probes
            total_batches: Total batches processed so far (for batch-based LR scheduling)
            last_lr_step_batch: Last batch number when LR was stepped
            last_validated_batch: Last batch number when validation was run
            validate_every_batches: Validate every N batches (None to disable)
            scheduler: Scheduler to step (if batch-based)
            scheduler_step_size_batches: Step size in batches for batch-based scheduler
            current_epoch: Current epoch number
            
        Returns:
            Tuple of updated training metrics and counters
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        print_interval = 1000
        batch_loss_sum = 0.0
        batch_count_since_print = 0
        last_print_time = time.time()
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            if batch_idx < skip_batches:
                continue

            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(features)
            loss = self._regression_loss_tensor(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            batch_loss_sum += loss.item()
            batch_count_since_print += 1
            
            # Print progress with average loss since last print (on same line)
            if batch_idx % print_interval == 0 and num_batches > 1:
                current_time = time.time()
                elapsed_time = current_time - last_print_time
                batches_per_second = batch_count_since_print / elapsed_time if elapsed_time > 0 else 0
                avg_loss_since_print = batch_loss_sum / batch_count_since_print
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx:5d}, Epoch {current_epoch}, Avg Loss: {avg_loss_since_print:.4f}, LR: {current_lr:.6f}, Speed: {batches_per_second:.1f} batches/s", end='\r', flush=True)
                batch_loss_sum = 0.0
                batch_count_since_print = 0
                last_print_time = current_time

            # Update learning rate (batch-based schedulers)
            if scheduler is not None and scheduler_step_size_batches is not None:
                current_total_batches = total_batches + batch_idx + 1
                batches_since_lr_step = current_total_batches - last_lr_step_batch
                if batches_since_lr_step >= scheduler_step_size_batches:
                    scheduler.step()
                    last_lr_step_batch = current_total_batches
                    if batch_idx % print_interval == 0:  # Only print if we're already printing
                        print(f" (LR reduced to {self.optimizer.param_groups[0]['lr']:.6f})", end='', flush=True)

            # Optional: validate every N batches
            if validate_every_batches is not None and validate_every_batches > 0 and val_loader is not None:
                current_total_batches = total_batches + batch_idx + 1
                batches_since_validation = current_total_batches - last_validated_batch
                if batches_since_validation >= validate_every_batches:
                    print()  # Newline before validation output
                    val_loss = self.validate(val_loader)
                    print(
                        f"Validation at batch {current_total_batches:,}: "
                        f"Val Loss: {val_loss:.4f}{self._format_last_val_metrics()}"
                    )
                    last_validated_batch = current_total_batches
                    running_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    self._append_learning_curve_point(
                        running_train_loss, val_loss, total_batches=current_total_batches
                    )
                    
                    # Update best model if validation improved
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._snapshot_run_state(
                            total_batches=current_total_batches,
                            last_saved_batch=last_saved_batch,
                            last_lr_step_batch=last_lr_step_batch,
                            last_validated_batch=current_total_batches,
                            epochs_without_improvement=0,
                            batches_in_epoch=batch_idx + 1,
                        )
                        self.save_model('best_model.pth')
                        print(f"  → New best model saved! (Val Loss: {val_loss:.4f})")
                        epochs_without_improvement = 0

            # Optional: save checkpoint every N batches during the epoch
            if save_every_batches is not None and save_every_batches > 0:
                current_total_batches = total_batches + batch_idx + 1
                batches_since_last_save = current_total_batches - last_saved_batch
                if batches_since_last_save >= save_every_batches:
                    last_saved_batch = current_total_batches
                    self._snapshot_run_state(
                        total_batches=current_total_batches,
                        last_saved_batch=last_saved_batch,
                        last_lr_step_batch=last_lr_step_batch,
                        last_validated_batch=last_validated_batch,
                        epochs_without_improvement=epochs_without_improvement,
                        batches_in_epoch=batch_idx + 1,
                    )
                    self.save_model(f'checkpoint_batch_{current_total_batches}.pth')
                    print(f"Saved checkpoint at {current_total_batches:,} batches")
            
            # Optional: run model probe every N iterations
            if probe_every_batches is not None and probe_every_batches > 0 and val_loader is not None:
                if batch_idx > 0 and (batch_idx % probe_every_batches == 0):
                    print()  # Newline before probe output
                    self.run_probe(val_loader)
        
        # Print newline at end of epoch to ensure clean output
        if num_batches > 0:
            print()  # Newline after epoch completes
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        current_total_batches = total_batches + skip_batches + num_batches
        self._snapshot_run_state(
            total_batches=current_total_batches,
            last_saved_batch=last_saved_batch,
            last_lr_step_batch=last_lr_step_batch,
            last_validated_batch=last_validated_batch,
            epochs_without_improvement=epochs_without_improvement,
            batches_in_epoch=0,
        )
        return avg_loss, num_batches, last_saved_batch, last_lr_step_batch, last_validated_batch, epochs_without_improvement
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_abs_cp = 0.0
        total_cp_elems = 0
        total_balanced_abs_cp = 0.0
        total_balanced_cp_elems = 0
        total_win_prob_bce = 0.0
        total_cp_regularizer = 0.0
        total_weighted_cp_regularizer = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss_parts = self._compute_loss_components(predictions, targets)
                loss = loss_parts['total']
                
                total_loss += loss.item()
                num_batches += 1
                y_cp = loss_parts['y_cp'].reshape(-1)
                yhat_cp = loss_parts['yhat_cp'].reshape(-1)
                total_abs_cp += (yhat_cp - y_cp).abs().sum().item()
                total_cp_elems += int(y_cp.numel())
                balanced_mask = y_cp.abs() <= self.balanced_eval_threshold_cp
                if balanced_mask.any():
                    total_balanced_abs_cp += (yhat_cp[balanced_mask] - y_cp[balanced_mask]).abs().sum().item()
                    total_balanced_cp_elems += int(balanced_mask.sum().item())
                total_win_prob_bce += float(loss_parts['win_prob_bce'].item())
                total_cp_regularizer += float(loss_parts['cp_regularizer'].item())
                total_weighted_cp_regularizer += float(loss_parts['weighted_cp_regularizer'].item())
        
        avg_loss = total_loss / num_batches
        if total_cp_elems > 0:
            self._last_val_cp_mae = total_abs_cp / total_cp_elems
        else:
            self._last_val_cp_mae = None
        if total_balanced_cp_elems > 0:
            self._last_val_balanced_cp_mae = total_balanced_abs_cp / total_balanced_cp_elems
        else:
            self._last_val_balanced_cp_mae = None
        self._last_val_metrics = {
            'total': avg_loss,
            'win_prob_bce': (total_win_prob_bce / num_batches) if num_batches > 0 else None,
            'cp_regularizer': (total_cp_regularizer / num_batches) if num_batches > 0 else None,
            'weighted_cp_regularizer': (total_weighted_cp_regularizer / num_batches) if num_batches > 0 else None,
            'cp_mae': self._last_val_cp_mae,
            'balanced_cp_mae': self._last_val_balanced_cp_mae,
            'balanced_threshold_cp': self.balanced_eval_threshold_cp,
        }
        if was_training:
            self.model.train()
        return avg_loss

    def run_probe(self, val_loader: DataLoader, max_batches: int = 1) -> None:
        """
        Run model probes to inspect model behavior on specific chess positions.
        """
        from trainer.model_probes_nnue import ModelProbeSuite
        from trainer.nnue_model import NNUEFeatureExtractor
        
        # Create feature extractor
        feature_extractor = NNUEFeatureExtractor()
        
        # Create probe suite
        probe_suite = ModelProbeSuite(cp_target_scale=self.cp_target_scale)
        
        # Run all probes
        results = probe_suite.run_all_probes(self.model, feature_extractor, self.device)
        
        # Print results in the exact format
        probe_suite.print_results(results)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              save_every: int = 10,
              save_every_batches: Optional[int] = 100000,
              validate_every_batches: Optional[int] = None,
              early_stopping_patience: int = 15,
              print_every: int = 1,
              probe_every_batches: Optional[int] = None,
              show_learning_curve: bool = False) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_every: Save model every N epochs
            save_every_batches: Save model every N batches (None to disable)
            validate_every_batches: Validate every N batches (None to disable, still validates at end of each epoch)
            early_stopping_patience: Stop training if no improvement for N epochs
            print_every: Print progress every N epochs
            probe_every_batches: Run model probes every N batches
            show_learning_curve: If True, plot train/val loss after each validation (requires matplotlib)
            
        Returns:
            Training history dictionary
        """
        if self.optimizer is None:
            raise ValueError("Must setup optimizer before training")
        
        self.show_learning_curve = bool(show_learning_curve)
        _bs = getattr(train_loader, "batch_size", None)
        self._lc_batch_size = int(_bs) if _bs is not None else 1
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {count_parameters(self.model):,} parameters")
        if self.regression_loss_name == "win_prob_bce":
            loss_desc = (
                f"Loss: win_prob_bce (soft BCE on White win prob, "
                f"sigma(cp/{self.win_prob_scale:g}))"
            )
            if self.cp_regularization_weight > 0:
                loss_desc += (
                    f" + {self.cp_regularization_weight:g} * "
                    f"{self._format_cp_regularizer_name()}"
                )
            print(loss_desc)
        else:
            print(f"Loss: {self.regression_loss_name}")
        if self.regression_loss_name == "win_prob_bce" and self.cp_regularization_weight > 0:
            print(
                f"Balanced validation slice: |target_cp| <= {self.balanced_eval_threshold_cp:g}"
            )
        if save_every_batches is not None:
            print(f"Will save checkpoint every {save_every_batches:,} batches")
        if validate_every_batches is not None:
            print(f"Will validate every {validate_every_batches:,} batches")
        
        start_time = time.time()
        epochs_without_improvement = int(self.run_state.get('epochs_without_improvement', 0))
        total_batches = int(self.run_state.get('total_batches', 0))
        last_saved_batch = int(self.run_state.get('last_saved_batch', 0))
        last_lr_step_batch = int(self.run_state.get('last_lr_step_batch', 0))
        last_validated_batch = int(self.run_state.get('last_validated_batch', 0))
        start_epoch = int(self.epoch)
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            skip_batches = self.resume_skip_batches if epoch == start_epoch else 0
            total_batches_before_epoch = total_batches - skip_batches
            
            # Training
            train_loss, batches_this_epoch, last_saved_batch, last_lr_step_batch, last_validated_batch, epochs_without_improvement = self.train_epoch(
                train_loader, 
                probe_every_batches=probe_every_batches, 
                val_loader=val_loader,
                total_batches=total_batches_before_epoch,
                last_saved_batch=last_saved_batch,
                last_lr_step_batch=last_lr_step_batch,
                last_validated_batch=last_validated_batch,
                validate_every_batches=validate_every_batches,
                save_every_batches=save_every_batches,
                scheduler=self.scheduler if hasattr(self, 'scheduler_step_on_batches') and self.scheduler_step_on_batches else None,
                scheduler_step_size_batches=self.scheduler_step_size_batches if hasattr(self, 'scheduler_step_size_batches') else None,
                current_epoch=epoch,
                epochs_without_improvement=epochs_without_improvement,
                skip_batches=skip_batches,
            )
            total_batches = total_batches_before_epoch + skip_batches + batches_this_epoch
            self.resume_skip_batches = 0
            self._record_lc_epoch_boundary(total_batches)
            
            # Validation (always at end of epoch)
            val_loss = self.validate(val_loader)
            
            # Update learning rate (epoch-based schedulers)
            if self.scheduler is not None and not (hasattr(self, 'scheduler_step_on_batches') and self.scheduler_step_on_batches):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Track history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            self.training_history['val_win_prob_bce'].append(self._last_val_metrics.get('win_prob_bce'))
            self.training_history['val_cp_regularizer'].append(self._last_val_metrics.get('cp_regularizer'))
            self.training_history['val_weighted_cp_regularizer'].append(
                self._last_val_metrics.get('weighted_cp_regularizer')
            )
            self.training_history['val_cp_mae'].append(self._last_val_cp_mae)
            self.training_history['val_balanced_cp_mae'].append(self._last_val_balanced_cp_mae)
            self._append_learning_curve_point(train_loss, val_loss, total_batches=total_batches)
            
            # Print progress
            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}{self._format_last_val_metrics()}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                      f"Time: {elapsed:.1f}s")
            
            # Save best model (when validation improves)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._snapshot_run_state(
                    total_batches=total_batches,
                    last_saved_batch=last_saved_batch,
                    last_lr_step_batch=last_lr_step_batch,
                    last_validated_batch=last_validated_batch,
                    epochs_without_improvement=0,
                    batches_in_epoch=0,
                )
                self.save_model('best_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint every N epochs
            if epoch % save_every == 0:
                self._snapshot_run_state(
                    total_batches=total_batches,
                    last_saved_batch=last_saved_batch,
                    last_lr_step_batch=last_lr_step_batch,
                    last_validated_batch=last_validated_batch,
                    epochs_without_improvement=epochs_without_improvement,
                    batches_in_epoch=0,
                )
                self.save_model(f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

            self._snapshot_run_state(
                total_batches=total_batches,
                last_saved_batch=last_saved_batch,
                last_lr_step_batch=last_lr_step_batch,
                last_validated_batch=last_validated_batch,
                epochs_without_improvement=epochs_without_improvement,
                batches_in_epoch=0,
            )
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.training_history
    
    def save_model(self, filename: str):
        """Save the model to disk."""
        filepath = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'run_state': self.run_state,
            'regression_loss': self.regression_loss_name,
            'cp_target_scale': self.cp_target_scale,
            'win_prob_scale': self.win_prob_scale,
            'cp_regularization_weight': self.cp_regularization_weight,
            'cp_regularization_loss': self.cp_regularization_loss_name,
            'cp_regularization_beta': self.cp_regularization_beta,
            'balanced_eval_threshold_cp': self.balanced_eval_threshold_cp,
            'last_val_metrics': self._last_val_metrics,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'val_win_prob_bce': [],
            'val_cp_regularizer': [],
            'val_weighted_cp_regularizer': [],
            'val_cp_mae': [],
            'val_balanced_cp_mae': [],
        })
        self.run_state.update(checkpoint.get('run_state', {}))
        self._last_val_metrics = checkpoint.get('last_val_metrics', {})
        self._last_val_cp_mae = self._last_val_metrics.get('cp_mae')
        self._last_val_balanced_cp_mae = self._last_val_metrics.get('balanced_cp_mae')
        saved_batches_in_epoch = int(self.run_state.get('batches_in_epoch', 0))
        if saved_batches_in_epoch > 0:
            self.resume_skip_batches = saved_batches_in_epoch
        else:
            self.epoch = int(self.epoch) + 1
            self.resume_skip_batches = 0
        
        print(f"Model loaded: {filepath}")


def train_nnue_model(data_file: str,
                     config: Dict[str, Any],
                     training_config: Dict[str, Any],
                     data_config: Optional[Dict[str, Any]] = None,
                     nnue: Optional[Dict[str, Any]] = None,
                     save_dir: str = 'checkpoints_nnue') -> NNUETrainer:
    """
    Train a NNUE model with the given configuration.
    
    Args:
        data_file: Path to training data file
        config: Model configuration
        training_config: Training configuration
        data_config: Data-loading configuration
        save_dir: Directory to save checkpoints
        
    Returns:
        Trained NNUE trainer
    """
    # Create model
    model = create_nnue_model(config)
    print(f"Created NNUE model with {count_parameters(model):,} parameters")
    
    nnue = nnue or {}
    trainer = NNUETrainer(
        model,
        save_dir=save_dir,
        regression_loss=str(nnue.get('regression_loss', 'smooth_l1')),
        huber_beta=float(nnue.get('huber_beta', 100.0)),
        cp_target_scale=float(nnue.get('cp_target_scale', 1.0)),
        win_prob_scale=float(nnue.get('win_prob_scale', 400.0)),
        cp_regularization_weight=float(nnue.get('cp_regularization_weight', 0.0)),
        cp_regularization_loss=str(nnue.get('cp_regularization_loss', 'smooth_l1')),
        cp_regularization_beta=float(nnue.get('cp_regularization_beta', 100.0)),
        balanced_eval_threshold_cp=float(nnue.get('balanced_eval_threshold_cp', 100.0)),
    )
    
    # Setup optimizer
    trainer.setup_optimizer(
        optimizer_type=training_config.get('optimizer', 'adamw'),
        learning_rate=training_config.get('learning_rate', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4),
        betas=tuple(training_config.get('betas', [0.9, 0.999])),
    )
    trainer.gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)
    
    # Setup scheduler
    trainer.setup_scheduler(
        scheduler_type=training_config.get('scheduler', 'step'),
        step_size=training_config.get('step_size', 10),
        gamma=training_config.get('gamma', 0.1)
    )
    
    data_config = data_config or {}
    train_loader, val_loader = NNUEDataLoader.create_data_loaders(
        train_file=data_file,
        val_file=data_file.replace('.train.', '.val.'),
        batch_size=training_config.get('batch_size', 32),
        max_train_positions=data_config.get('max_train_positions'),
        max_val_positions=data_config.get('max_val_positions'),
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', True),
        cp_label_clip_min=float(nnue.get('cp_label_clip_min', -2000.0)),
        cp_label_clip_max=float(nnue.get('cp_label_clip_max', 2000.0)),
        cp_target_scale=float(nnue.get('cp_target_scale', 1.0)),
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config.get('num_epochs', 100),
        save_every=training_config.get('save_every', 10),
        early_stopping_patience=training_config.get('early_stopping_patience', 15),
        show_learning_curve=training_config.get('show_learning_curve', False),
    )
    
    return trainer


if __name__ == "__main__":
    # Test the NNUE trainer
    print("Testing NNUE trainer...")
    
    # Create a small test model
    config = {
        'hidden_sizes': [64],  # Single hidden layer with 64 neurons
        'dropout': 0.1
    }
    
    model = create_nnue_model(config)
    trainer = NNUETrainer(model)
    
    print(f"NNUE trainer created with {count_parameters(model):,} parameters")
    print("NNUE trainer test completed successfully!")
