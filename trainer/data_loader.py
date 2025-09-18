"""
Data Loading Pipeline for Chess Training

This module provides efficient data loading for chess position training data,
including support for compressed .zst files and streaming batch processing.
"""

import json
import zstandard as zstd
import torch
import chess
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, Dict, Any, List, Optional, Tuple
import os
import sys
from pathlib import Path

# Add parent directory to path to import chess_game modules
sys.path.append(str(Path(__file__).parent.parent))

from chess_game.neural_network_evaluator import NeuralNetworkEvaluator


class ChessPositionDataset(Dataset):
    """
    PyTorch Dataset for chess position training data.
    
    Loads data from compressed .zst files and provides streaming access
    to chess positions with their evaluations.
    """
    
    def __init__(self, 
                 data_file: str,
                 max_positions: Optional[int] = None,
                 shuffle: bool = True,
                 cache_size: int = 10000):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to the compressed .zst data file
            max_positions: Maximum number of positions to load (None for all)
            shuffle: Whether to shuffle the data
            cache_size: Number of positions to keep in memory cache
        """
        self.data_file = data_file
        self.max_positions = max_positions
        self.shuffle = shuffle
        self.cache_size = cache_size
        
        # Initialize feature extractor
        self.feature_extractor = NeuralNetworkEvaluator()
        
        # Load and cache data
        self.positions = []
        self.evaluations = []
        self._load_data()
        
        # Create indices for shuffling
        self.indices = list(range(len(self.positions)))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_data(self):
        """Load data from the compressed .zst file."""
        print(f"Loading data from {self.data_file}...")
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        if not self.data_file.endswith('.zst'):
            raise ValueError(f"Expected .zst compressed file, got: {self.data_file}")
        
        # Open compressed file
        with open(self.data_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(f)
            buffer = ""
            position_count = 0
            
            while True:
                chunk = reader.read(8192)  # Read in chunks
                if not chunk:
                    break
                
                buffer += chunk.decode('utf-8', errors='ignore')
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:  # Process complete lines
                    if line.strip():
                        try:
                            data = json.loads(line)
                            position, evaluation = self._parse_position_data(data)
                            
                            if position is not None and evaluation is not None:
                                self.positions.append(position)
                                self.evaluations.append(evaluation)
                                position_count += 1
                                
                                if self.max_positions and position_count >= self.max_positions:
                                    break
                                    
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            # Skip malformed lines
                            continue
                
                if self.max_positions and position_count >= self.max_positions:
                    break
            
            # Process any remaining data in buffer
            if buffer.strip():
                try:
                    data = json.loads(buffer)
                    position, evaluation = self._parse_position_data(data)
                    if position is not None and evaluation is not None:
                        self.positions.append(position)
                        self.evaluations.append(evaluation)
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        
        print(f"Loaded {len(self.positions)} positions from {self.data_file}")
    
    def _parse_position_data(self, data: Dict[str, Any]) -> Tuple[Optional[chess.Board], Optional[float]]:
        """
        Parse a single position from the JSON data.
        
        Args:
            data: JSON data containing FEN and evaluations
            
        Returns:
            Tuple of (board, evaluation) or (None, None) if parsing fails
        """
        try:
            # Extract FEN
            fen = data['fen']
            board = chess.Board(fen)
            
            # Extract evaluation from the first entry in evals
            evals = data['evals']
            if not evals:
                return None, None
            
            # Get the first evaluation entry
            first_eval = evals[0]
            pvs = first_eval['pvs']
            if not pvs:
                return None, None
            
            # Get the primary variation evaluation
            primary_pv = pvs[0]
            
            # Handle both cp and mate evaluations
            if 'cp' in primary_pv:
                cp_eval = primary_pv['cp']
                # Convert to side-to-move perspective
                if not board.turn:  # Black to move
                    cp_eval = -cp_eval
                return board, float(cp_eval)
            elif 'mate' in primary_pv:
                mate_eval = primary_pv['mate']
                # Convert mate to large cp value
                if mate_eval > 0:  # White mates
                    cp_eval = 10000 - mate_eval
                else:  # Black mates
                    cp_eval = -10000 - mate_eval
                
                # Convert to side-to-move perspective
                if not board.turn:  # Black to move
                    cp_eval = -cp_eval
                return board, float(cp_eval)
            else:
                return None, None
                
        except (KeyError, ValueError, chess.InvalidMoveError) as e:
            return None, None
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tuple of (features, target) where features is the board representation
            and target is the win probability
        """
        actual_idx = self.indices[idx]
        board = self.positions[actual_idx]
        cp_eval = self.evaluations[actual_idx]
        
        # Convert board to features using the existing feature extractor
        features = self.feature_extractor._board_to_features(board)
        # Convert from (8, 8, 18) to (18, 8, 8) for PyTorch conv layers
        features = np.transpose(features, (2, 0, 1))
        features_tensor = torch.from_numpy(features).float()
        
        # Convert cp evaluation to win probability
        win_prob = self._cp_to_win_probability(cp_eval)
        target_tensor = torch.tensor([win_prob], dtype=torch.float32)
        
        return features_tensor, target_tensor
    
    def _cp_to_win_probability(self, cp: float) -> float:
        """
        Convert centipawn evaluation to win probability using logistic function.
        
        The logistic function only saturates if cp is outside [-5000, 5000].
        
        Args:
            cp: Centipawn evaluation from side-to-move perspective
            
        Returns:
            Win probability in range [0, 1]
        """
        # Clamp cp to [-5000, 5000] to prevent saturation
        cp_clamped = np.clip(cp, -5000, 5000)
        
        # Use logistic function: 1 / (1 + exp(-cp/400))
        # The 400 scaling factor is commonly used in chess engines
        win_prob = 1.0 / (1.0 + np.exp(-cp_clamped / 400.0))
        
        return float(win_prob)


class ChessDataLoader:
    """
    High-level data loader for chess training data.
    
    Provides convenient methods for creating data loaders with different configurations.
    """
    
    @staticmethod
    def create_dataloader(data_file: str,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         max_positions: Optional[int] = None,
                         pin_memory: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader for chess training data.
        
        Args:
            data_file: Path to the compressed .zst data file
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            max_positions: Maximum number of positions to load
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            Configured DataLoader
        """
        dataset = ChessPositionDataset(
            data_file=data_file,
            max_positions=max_positions,
            shuffle=shuffle
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop incomplete batches
        )
        
        return dataloader
    
    @staticmethod
    def create_train_val_loaders(data_file: str,
                                batch_size: int = 32,
                                val_split: float = 0.1,
                                num_workers: int = 4,
                                max_positions: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.
        
        Args:
            data_file: Path to the compressed .zst data file
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            num_workers: Number of worker processes
            max_positions: Maximum number of positions to load
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load full dataset
        full_dataset = ChessPositionDataset(
            data_file=data_file,
            max_positions=max_positions,
            shuffle=True
        )
        
        # Split into train and validation
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader


def test_data_loader():
    """Test function for the data loader."""
    # Test with compressed sample data
    sample_file = "/Users/rajat/chess-game/trainer/localdata/lichess_db_eval.sample.jsonl.zst"
    
    if os.path.exists(sample_file):
        print("Testing data loader with compressed sample data...")
        
        # Create dataset
        dataset = ChessPositionDataset(sample_file, max_positions=100)
        print(f"Dataset size: {len(dataset)}")
        
        # Test single item
        if len(dataset) > 0:
            features, target = dataset[0]
            print(f"Features shape: {features.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Target value: {target.item():.4f}")
        
        # Test data loader
        dataloader = ChessDataLoader.create_dataloader(
            sample_file, 
            batch_size=4, 
            max_positions=20
        )
        
        print("\nTesting batch loading...")
        for i, (batch_features, batch_targets) in enumerate(dataloader):
            print(f"Batch {i}: features {batch_features.shape}, targets {batch_targets.shape}")
            if i >= 2:  # Test first few batches
                break
    else:
        print(f"Compressed sample file not found: {sample_file}")


if __name__ == "__main__":
    test_data_loader()
