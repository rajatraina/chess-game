"""
Streaming Data Loading Pipeline for Chess Training

This module provides memory-efficient streaming data loading for chess position training data,
reading positions on-demand instead of loading everything into memory.
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
import random
from pathlib import Path

# Add parent directory to path to import chess_game modules
sys.path.append(str(Path(__file__).parent.parent))

from chess_game.neural_network_evaluator import NeuralNetworkEvaluator


class StreamingChessPositionDataset(Dataset):
    """
    Streaming PyTorch Dataset for chess position training data.
    
    Reads data on-demand from compressed .zst files without loading everything into memory.
    Uses a buffer-based approach for efficient streaming.
    """
    
    def __init__(self, 
                 data_file: str,
                 max_positions: Optional[int] = None,
                 shuffle: bool = True,
                 buffer_size: int = 10000,
                 prefetch_factor: int = 2):
        """
        Initialize the streaming dataset.
        
        Args:
            data_file: Path to the compressed .zst data file
            max_positions: Maximum number of positions to use (None for all)
            shuffle: Whether to shuffle the data
            buffer_size: Number of positions to keep in memory buffer
            prefetch_factor: How many batches to prefetch ahead
        """
        self.data_file = data_file
        self.max_positions = max_positions
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        
        # Initialize feature extractor
        self.feature_extractor = NeuralNetworkEvaluator()
        
        # Streaming state
        self.file_handle = None
        self.decompressor = None
        self.reader = None
        self.buffer = ""
        self.position_buffer = []
        self.evaluation_buffer = []
        self.positions_loaded = 0
        self.positions_used = 0
        self.file_exhausted = False
        
        # Initialize streaming
        self._init_streaming()
        
        # Pre-fill buffer
        self._fill_buffer()
    
    def _init_streaming(self):
        """Initialize the streaming file reader."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        if not self.data_file.endswith('.zst'):
            raise ValueError(f"Expected .zst compressed file, got: {self.data_file}")
        
        self.file_handle = open(self.data_file, 'rb')
        self.decompressor = zstd.ZstdDecompressor()
        self.reader = self.decompressor.stream_reader(self.file_handle)
        
        print(f"Initialized streaming from {self.data_file}")
    
    def _fill_buffer(self):
        """Fill the position buffer with new data from the file."""
        if self.file_exhausted:
            return
        
        # Read data in chunks until buffer is full or file is exhausted
        while len(self.position_buffer) < self.buffer_size and not self.file_exhausted:
            try:
                chunk = self.reader.read(8192)  # Read in chunks
                if not chunk:
                    self.file_exhausted = True
                    break
                
                self.buffer += chunk.decode('utf-8', errors='ignore')
                lines = self.buffer.split('\n')
                self.buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:  # Process complete lines
                    if line.strip():
                        try:
                            data = json.loads(line)
                            position, evaluation = self._parse_position_data(data)
                            
                            if position is not None and evaluation is not None:
                                self.position_buffer.append(position)
                                self.evaluation_buffer.append(evaluation)
                                self.positions_loaded += 1
                                
                                # Check if we've reached max_positions
                                if self.max_positions and self.positions_loaded >= self.max_positions:
                                    self.file_exhausted = True
                                    break
                                    
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # Skip malformed lines
                            continue
                
            except Exception as e:
                print(f"Error reading from file: {e}")
                self.file_exhausted = True
                break
        
        # Process any remaining data in buffer
        if self.buffer.strip() and not self.file_exhausted:
            try:
                data = json.loads(self.buffer)
                position, evaluation = self._parse_position_data(data)
                if position is not None and evaluation is not None:
                    self.position_buffer.append(position)
                    self.evaluation_buffer.append(evaluation)
                    self.positions_loaded += 1
                self.buffer = ""
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        
        # Shuffle buffer if requested
        if self.shuffle and len(self.position_buffer) > 1:
            combined = list(zip(self.position_buffer, self.evaluation_buffer))
            random.shuffle(combined)
            self.position_buffer, self.evaluation_buffer = zip(*combined)
            self.position_buffer = list(self.position_buffer)
            self.evaluation_buffer = list(self.evaluation_buffer)
    
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
                
        except (KeyError, ValueError, chess.InvalidMoveError):
            return None, None
    
    def __len__(self) -> int:
        """Return the total number of positions available."""
        if self.max_positions:
            return min(self.max_positions, self.positions_loaded + (self.buffer_size if not self.file_exhausted else 0))
        else:
            # For streaming, we can't know the exact length
            # Return a large number to indicate "many"
            return 1000000  # This is just for PyTorch compatibility
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example (ignored in streaming mode)
            
        Returns:
            Tuple of (features, target) where features is the board representation
            and target is the win probability
        """
        # If buffer is empty or nearly empty, refill it
        if len(self.position_buffer) < 10:  # Refill when buffer gets low
            self._fill_buffer()
        
        # If no data available, raise error
        if not self.position_buffer:
            raise IndexError("No more data available")
        
        # Get next position from buffer
        board = self.position_buffer.pop(0)
        cp_eval = self.evaluation_buffer.pop(0)
        self.positions_used += 1
        
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
        
        Args:
            cp: Centipawn evaluation from side-to-move perspective
            
        Returns:
            Win probability in range [0, 1]
        """
        # Clamp cp to [-5000, 5000] to prevent saturation
        cp_clamped = np.clip(cp, -5000, 5000)
        
        # Use logistic function: 1 / (1 + exp(-cp/400))
        win_prob = 1.0 / (1.0 + np.exp(-cp_clamped / 400.0))
        
        return float(win_prob)
    
    def __del__(self):
        """Clean up file handles."""
        if self.file_handle:
            self.file_handle.close()


class StreamingChessDataLoader:
    """
    High-level streaming data loader for chess training data.
    
    Provides convenient methods for creating streaming data loaders.
    """
    
    @staticmethod
    def create_dataloader(data_file: str,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 0,  # Must be 0 for streaming
                         max_positions: Optional[int] = None,
                         buffer_size: int = 10000,
                         pin_memory: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader for streaming chess training data.
        
        Args:
            data_file: Path to the compressed .zst data file
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes (must be 0 for streaming)
            max_positions: Maximum number of positions to load
            buffer_size: Size of the streaming buffer
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            Configured DataLoader
        """
        if num_workers > 0:
            print("Warning: num_workers must be 0 for streaming data loader")
            num_workers = 0
        
        dataset = StreamingChessPositionDataset(
            data_file=data_file,
            max_positions=max_positions,
            shuffle=shuffle,
            buffer_size=buffer_size
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffling is handled by the dataset
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop incomplete batches
        )
        
        return dataloader
    
    @staticmethod
    def create_train_val_loaders(data_file: str,
                                batch_size: int = 32,
                                val_split: float = 0.1,
                                num_workers: int = 0,
                                max_positions: Optional[int] = None,
                                buffer_size: int = 10000) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders with streaming.
        
        Note: For streaming, we create separate datasets for train/val
        by using different max_positions limits.
        
        Args:
            data_file: Path to the compressed .zst data file
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            num_workers: Number of worker processes (must be 0)
            max_positions: Maximum number of positions to load
            buffer_size: Size of the streaming buffer
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if num_workers > 0:
            print("Warning: num_workers must be 0 for streaming data loader")
            num_workers = 0
        
        # Calculate train/val split
        if max_positions:
            train_positions = int(max_positions * (1 - val_split))
            val_positions = max_positions - train_positions
        else:
            # For unlimited data, use a large number
            train_positions = int(1000000 * (1 - val_split))  # 900k for training
            val_positions = 100000  # 100k for validation
        
        # Create training dataset
        train_dataset = StreamingChessPositionDataset(
            data_file=data_file,
            max_positions=train_positions,
            shuffle=True,
            buffer_size=buffer_size
        )
        
        # Create validation dataset (starts from where training left off)
        val_dataset = StreamingChessPositionDataset(
            data_file=data_file,
            max_positions=val_positions,
            shuffle=False,  # No need to shuffle validation data
            buffer_size=buffer_size // 2  # Smaller buffer for validation
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffling handled by dataset
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


def test_streaming_data_loader():
    """Test function for the streaming data loader."""
    # Test with compressed sample data
    sample_file = "/Users/rajat/chess-game/trainer/localdata/lichess_db_eval.sample.jsonl.zst"
    
    if os.path.exists(sample_file):
        print("Testing streaming data loader with compressed sample data...")
        
        # Create dataset
        dataset = StreamingChessPositionDataset(sample_file, max_positions=100, buffer_size=20)
        print(f"Dataset initialized for streaming")
        
        # Test single items
        for i in range(5):
            try:
                features, target = dataset[i]
                print(f"Item {i}: features {features.shape}, target {target.item():.4f}")
            except IndexError:
                print(f"No more data available after {i} items")
                break
        
        # Test data loader
        dataloader = StreamingChessDataLoader.create_dataloader(
            sample_file, 
            batch_size=4, 
            max_positions=20,
            buffer_size=10
        )
        
        print("\nTesting batch loading...")
        for i, (batch_features, batch_targets) in enumerate(dataloader):
            print(f"Batch {i}: features {batch_features.shape}, targets {batch_targets.shape}")
            if i >= 2:  # Test first few batches
                break
    else:
        print(f"Compressed sample file not found: {sample_file}")


if __name__ == "__main__":
    test_streaming_data_loader()
