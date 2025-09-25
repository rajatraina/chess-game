"""
Streaming Data Loading Pipeline for Chess Training

This module provides memory-efficient streaming data loading for chess position training data,
using an iterator-based approach that works with PyTorch's DataLoader.
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


class ChessIterator:
    """
    Iterator for streaming chess positions from compressed files.
    
    This class handles the low-level streaming from the .zst file and provides
    an iterator interface that can be used with PyTorch datasets.
    """
    
    def __init__(self, 
                 data_file: str,
                 max_positions: Optional[int] = None,
                 buffer_size: int = 10000,
                 start_offset: int = 0,
                 is_validation: bool = False,
                 val_set_size: int = 10000,
                 feature_extractor: Optional[NeuralNetworkEvaluator] = None,
                 cp_to_prob_scale: float = 400.0):
        """
        Initialize the streaming iterator.
        
        Args:
            data_file: Path to the compressed .zst data file
            max_positions: Maximum number of positions to yield
            buffer_size: Number of positions to keep in memory buffer
            start_offset: Number of positions to skip from the beginning of the file
            is_validation: Whether this iterator is for validation (collects first N positions)
            val_set_size: Number of positions to use for validation (first N positions)
            feature_extractor: Shared feature extractor (optional)
            cp_to_prob_scale: Scale factor for centipawn to probability conversion
        """
        self.data_file = data_file
        self.max_positions = max_positions
        self.buffer_size = buffer_size
        self.start_offset = start_offset
        self.is_validation = is_validation
        self.val_set_size = val_set_size
        self.cp_to_prob_scale = cp_to_prob_scale
        
        # Use shared feature extractor or create new one
        self.feature_extractor = feature_extractor or NeuralNetworkEvaluator()
        
        # Streaming state
        self.file_handle = None
        self.decompressor = None
        self.reader = None
        self.buffer = ""
        self.position_buffer = []
        self.evaluation_buffer = []
        self.fen_buffer = []  # Store FEN strings for validation logging
        self.positions_yielded = 0
        self.positions_skipped = 0  # Track positions skipped for start_offset
        self.position_counter = 0  # Track absolute position number for val_every_n logic
        self.file_exhausted = False
        
        # Initialize streaming
        self._init_streaming()
    
    def _init_streaming(self):
        """Initialize the streaming file reader."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        if not self.data_file.endswith('.zst'):
            raise ValueError(f"Expected .zst compressed file, got: {self.data_file}")
        
        self.file_handle = open(self.data_file, 'rb')
        self.decompressor = zstd.ZstdDecompressor()
        self.reader = self.decompressor.stream_reader(self.file_handle)
        
        # Only print initialization message once per dataset
        if not hasattr(ChessIterator, '_initialization_printed'):
            print(f"Initialized streaming iterator from {self.data_file}")
            ChessIterator._initialization_printed = True
    
    def _fill_buffer(self):
        """Fill the position buffer with new data from the file."""
        if self.file_exhausted:
            return
        
        # Error tracking for data quality monitoring
        total_lines_processed = 0
        error_count = 0
        max_error_percentage = 1.0  # Warn if more than 1% of lines have errors
        
        # For validation datasets, stop when we have enough positions
        target_size = self.buffer_size
        if self.is_validation and self.val_set_size:
            target_size = min(self.buffer_size, self.val_set_size)
        
        # Read data in chunks until buffer is full or file is exhausted
        while len(self.position_buffer) < target_size and not self.file_exhausted:
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
                        total_lines_processed += 1
                        try:
                            data = json.loads(line)
                            position, evaluation, fen = self._parse_position_data(data)
                            
                            if position is not None and evaluation is not None:
                                # Increment position counter
                                self.position_counter += 1
                                
                                # Skip positions until we reach start_offset
                                if self.positions_skipped < self.start_offset:
                                    self.positions_skipped += 1
                                    continue
                                
                                # Route position based on iterator type and position number
                                if self.is_validation:
                                    # Validation iterator: collect first N positions
                                    if self.position_counter <= self.val_set_size:
                                        self.position_buffer.append(position)
                                        self.evaluation_buffer.append(evaluation)
                                        self.fen_buffer.append(fen)
                                else:
                                    # Training iterator: collect positions after validation set
                                    # If val_set_size is larger than max_positions, use all positions for training
                                    if self.val_set_size >= (self.max_positions or float('inf')):
                                        # All positions go to training if validation set is larger than dataset
                                        self.position_buffer.append(position)
                                        self.evaluation_buffer.append(evaluation)
                                        self.fen_buffer.append(fen)
                                    elif self.position_counter > self.val_set_size:
                                        self.position_buffer.append(position)
                                        self.evaluation_buffer.append(evaluation)
                                        self.fen_buffer.append(fen)
                                
                                # Check if we've reached max_positions
                                if self.max_positions and len(self.position_buffer) >= self.max_positions:
                                    self.file_exhausted = True
                                    break
                                    
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # Track errors for quality monitoring
                            error_count += 1
                            continue
                
            except Exception as e:
                print(f"Error reading from file: {e}")
                self.file_exhausted = True
                break
        
        # Report data quality statistics
        if total_lines_processed > 0:
            error_percentage = (error_count / total_lines_processed) * 100
            if error_percentage > max_error_percentage:
                print(f"⚠️  High error rate: {error_percentage:.1f}% of lines had errors "
                      f"({error_count}/{total_lines_processed} lines). "
                      f"Buffer filled with {len(self.position_buffer)} positions.")
            elif error_count > 0:
                print(f"📊 Data quality: {error_percentage:.1f}% error rate "
                      f"({error_count}/{total_lines_processed} lines). "
                      f"Buffer filled with {len(self.position_buffer)} positions.")
        
        # Process any remaining data in buffer
        if self.buffer.strip() and not self.file_exhausted:
            try:
                data = json.loads(self.buffer)
                position, evaluation, fen = self._parse_position_data(data)
                if position is not None and evaluation is not None:
                    self.position_buffer.append(position)
                    self.evaluation_buffer.append(evaluation)
                    self.fen_buffer.append(fen)
                self.buffer = ""
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        
        # No shuffling needed - input data is already shuffled
    
    def _parse_position_data(self, data: Dict[str, Any]) -> Tuple[Optional[chess.Board], Optional[float], Optional[str]]:
        """Parse a single position from the JSON data."""
        try:
            # Extract FEN
            fen = data['fen']
            try:
                board = chess.Board(fen)
            except (ValueError, TypeError) as e:
                # Skip malformed FEN strings
                return None, None, None
            
            # Extract evaluation from the first entry in evals
            evals = data['evals']
            if not evals:
                return None, None, None
            
            # Get the first evaluation entry
            first_eval = evals[0]
            pvs = first_eval['pvs']
            if not pvs:
                return None, None, None
            
            # Get the primary variation evaluation
            primary_pv = pvs[0]
            
            # Handle both cp and mate evaluations
            if 'cp' in primary_pv:
                cp_eval = primary_pv['cp']
                # Convert to side-to-move perspective
                if not board.turn:  # Black to move
                    cp_eval = -cp_eval
                return board, float(cp_eval), fen
            elif 'mate' in primary_pv:
                mate_eval = primary_pv['mate']
                # Convert mate to cp value (2000 - 10 * distance to mate)
                if mate_eval > 0:  # White mates
                    cp_eval = 2000 - 10 * mate_eval
                else:  # Black mates
                    cp_eval = -2000 - 10 * mate_eval
                
                # Convert to side-to-move perspective
                if not board.turn:  # Black to move
                    cp_eval = -cp_eval
                return board, float(cp_eval), fen
            else:
                return None, None, None
                
        except (KeyError, ValueError, chess.InvalidMoveError):
            return None, None, None
    
    def _cp_to_win_probability(self, cp: float) -> float:
        """Convert centipawn evaluation to win probability using logistic function."""
        # Clamp cp to [-5000, 5000] to prevent saturation
        cp_clamped = np.clip(cp, -5000, 5000)
        
        # Use logistic function: 1 / (1 + exp(-cp/scale_factor))
        # Scale factor: 400 = moderate curve, 200 = steeper, 600 = gentler
        scale_factor = getattr(self, 'cp_to_prob_scale', 400.0)
        win_prob = 1.0 / (1.0 + np.exp(-cp_clamped / scale_factor))
        
        return float(win_prob)
    
    def __iter__(self):
        """Make this class iterable."""
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the next training example."""
        # Check if we've reached max_positions
        if self.max_positions and self.positions_yielded >= self.max_positions:
            raise StopIteration
        
        # If buffer is empty or nearly empty, refill it
        if len(self.position_buffer) < 10:
            self._fill_buffer()
        
        # If no data available, stop iteration
        if not self.position_buffer:
            raise StopIteration
        
        # Get next position from buffer
        board = self.position_buffer.pop(0)
        cp_eval = self.evaluation_buffer.pop(0)
        self.positions_yielded += 1
        
        # Convert board to features
        features = self.feature_extractor._board_to_features(board)
        # Convert from (8, 8, 18) to (18, 8, 8) for PyTorch conv layers
        features = np.transpose(features, (2, 0, 1))
        features_tensor = torch.from_numpy(features).float()
        
        # Convert cp evaluation to win probability
        win_prob = self._cp_to_win_probability(cp_eval)
        target_tensor = torch.tensor(win_prob, dtype=torch.float32)
        
        return features_tensor, target_tensor
    
    def get_random_positions(self, num_positions: int = 10) -> List[Tuple[str, float, torch.Tensor]]:
        """
        Get random positions with FEN, ground truth evaluation, and features for validation logging.
        
        Args:
            num_positions: Number of random positions to return
            
        Returns:
            List of tuples (fen, ground_truth_eval, features)
        """
        # Ensure we have data in the buffer by filling it if needed
        if not self.position_buffer or not self.evaluation_buffer or not self.fen_buffer:
            # Try to fill the buffer if it's empty
            if not self.file_exhausted:
                self._fill_buffer()
            
            # If still no data, return empty list
            if not self.position_buffer or not self.evaluation_buffer or not self.fen_buffer:
                return []
        
        # Get random indices
        available_positions = min(num_positions, len(self.position_buffer))
        random_indices = random.sample(range(len(self.position_buffer)), available_positions)
        
        results = []
        for idx in random_indices:
            fen = self.fen_buffer[idx]
            ground_truth_eval = self.evaluation_buffer[idx]
            
            # Get features for this position (same as training pipeline)
            features = self.feature_extractor._board_to_features(self.position_buffer[idx])
            # Convert from (8, 8, 18) to (18, 8, 8) for PyTorch conv layers
            features = np.transpose(features, (2, 0, 1))
            features = torch.from_numpy(features).float()
            
            results.append((fen, ground_truth_eval, features))
        
        return results
    
    def __del__(self):
        """Clean up file handles."""
        if self.file_handle:
            self.file_handle.close()


class ChessDataset(Dataset):
    """
    PyTorch Dataset that wraps the streaming iterator.
    
    This dataset provides a PyTorch-compatible interface while using
    the streaming iterator for memory-efficient data loading.
    """
    
    def __init__(self, 
                 data_file: str,
                 max_positions: Optional[int] = None,
                 buffer_size: int = 10000,
                 start_offset: int = 0,
                 is_validation: bool = False,
                 val_set_size: int = 10000,
                 cp_to_prob_scale: float = 400.0):
        """
        Initialize the streaming dataset.
        
        Args:
            data_file: Path to the compressed .zst data file
            max_positions: Maximum number of positions to use
            buffer_size: Size of the streaming buffer
            start_offset: Number of positions to skip from the beginning of the file
            is_validation: Whether this dataset is for validation
            val_set_size: Number of positions to use for validation (first N positions)
            cp_to_prob_scale: Scale factor for centipawn to probability conversion
        """
        self.data_file = data_file
        self.max_positions = max_positions
        self.buffer_size = buffer_size
        self.start_offset = start_offset
        self.is_validation = is_validation
        self.val_set_size = val_set_size
        self.cp_to_prob_scale = cp_to_prob_scale
        
        # Track current position in the iterator
        self.current_iterator = None
        
        # Create shared feature extractor (reused across epochs)
        self.feature_extractor = NeuralNetworkEvaluator()
        
        self.iterator_config = {
            'data_file': data_file,
            'max_positions': max_positions,
            'buffer_size': buffer_size,
            'start_offset': start_offset,
            'is_validation': is_validation,
            'val_set_size': val_set_size,
            'feature_extractor': self.feature_extractor,  # Pass shared extractor
            'cp_to_prob_scale': cp_to_prob_scale  # Pass scale factor
        }
    
    def __len__(self) -> int:
        """Return the total number of positions available."""
        if self.max_positions:
            if self.is_validation:
                # Validation dataset gets the first val_set_size positions
                return min(self.val_set_size, self.max_positions)
            else:
                # Training dataset gets positions after validation set
                # If val_set_size is larger than max_positions, use all positions for training
                if self.val_set_size >= self.max_positions:
                    return self.max_positions
                else:
                    return max(1, self.max_positions - self.val_set_size)
        else:
            # For unlimited data, return a large number
            return 1000000  # This is just for PyTorch compatibility
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example (ignored in streaming mode)
            
        Returns:
            Tuple of (features, target)
        """
        # Initialize iterator if needed
        if self.current_iterator is None:
            self.iterator = ChessIterator(**self.iterator_config)
            self.current_iterator = iter(self.iterator)
        
        try:
            return next(self.current_iterator)
        except StopIteration:
            # Reset iterator for next epoch - create new iterator to reset file position
            self.iterator = ChessIterator(**self.iterator_config)
            self.current_iterator = iter(self.iterator)
            return next(self.current_iterator)
    
    def get_random_positions(self, num_positions: int = 10) -> List[Tuple[str, float, torch.Tensor]]:
        """
        Get random positions with FEN, ground truth evaluation, and features for validation logging.
        
        Args:
            num_positions: Number of random positions to return
            
        Returns:
            List of tuples (fen, ground_truth_eval, features)
        """
        return self.iterator.get_random_positions(num_positions)


class ChessDataLoader:
    """
    High-level streaming data loader for chess training data.
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
        """
        if num_workers > 0:
            print("Warning: num_workers must be 0 for streaming data loader")
            num_workers = 0
        
        dataset = ChessDataset(
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
                                val_set_size: int = 10000,
                                num_workers: int = 0,
                                max_positions: Optional[int] = None,
                                buffer_size: int = 10000) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders with streaming.
        Uses first N positions for validation, rest for training.
        """
        if num_workers > 0:
            print("Warning: num_workers must be 0 for streaming data loader")
            num_workers = 0
        
        # Create training dataset (collects positions after validation set)
        train_dataset = ChessDataset(
            data_file=data_file,
            max_positions=max_positions,
            shuffle=True,
            buffer_size=buffer_size,
            is_validation=False,
            val_set_size=val_set_size
        )
        
        # Create validation dataset (collects first N positions)
        # Use smaller buffer size for validation to avoid hitting max attempts
        val_buffer_size = min(val_set_size * 2, buffer_size // 2)
        val_dataset = ChessDataset(
            data_file=data_file,
            max_positions=max_positions,
            shuffle=False,  # No need to shuffle validation data
            buffer_size=val_buffer_size,  # Appropriate buffer size for validation
            is_validation=True,
            val_set_size=val_set_size
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

    @staticmethod
    def create_separate_loaders(train_file: str,
                               val_file: str,
                               batch_size: int = 32,
                               num_workers: int = 0,
                               max_positions: Optional[int] = None,
                               buffer_batches: int = 20,
                               cp_to_prob_scale: float = 400.0) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders from separate files.
        
        Args:
            train_file: Path to training data file (streaming)
            val_file: Path to validation data file (cached)
            batch_size: Batch size for both loaders
            num_workers: Number of worker processes (must be 0 for streaming)
            max_positions: Maximum number of training positions
            buffer_batches: Number of batches to read in one go (buffer_size = batch_size × buffer_batches)
            cp_to_prob_scale: Scale factor for centipawn to probability conversion
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if num_workers > 0:
            print("Warning: num_workers must be 0 for streaming data loader")
            num_workers = 0
        
        # Calculate buffer size from batches
        buffer_size = batch_size * buffer_batches
        print(f"Buffer size: {buffer_size} positions ({buffer_batches} batches × {batch_size} batch_size)")
        
        # Create training dataset (streaming)
        train_dataset = ChessDataset(
            data_file=train_file,
            max_positions=max_positions,
            buffer_size=buffer_size,
            is_validation=False,
            val_set_size=0,  # Not used for separate files
            cp_to_prob_scale=cp_to_prob_scale
        )
        
        # Create validation dataset (cached)
        val_dataset = CachedValidationDataset(val_file, cp_to_prob_scale=cp_to_prob_scale)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Input data is already shuffled
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Drop incomplete batches
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False  # Keep all validation data
        )
        
        return train_loader, val_loader


class CachedValidationDataset(Dataset):
    """
    Cached validation dataset that loads all data into memory once.
    Used for small validation files that can fit in memory.
    """
    
    def __init__(self, data_file: str, feature_extractor: Optional[NeuralNetworkEvaluator] = None, cp_to_prob_scale: float = 400.0):
        """
        Initialize cached validation dataset.
        
        Args:
            data_file: Path to the validation data file
            feature_extractor: Shared feature extractor
            cp_to_prob_scale: Scale factor for centipawn to probability conversion
        """
        self.data_file = data_file
        self.feature_extractor = feature_extractor or NeuralNetworkEvaluator()
        self.cp_to_prob_scale = cp_to_prob_scale
        
        # Load all data into memory
        print(f"Loading validation data from {data_file}...")
        self.positions = []
        self.evaluations = []
        self.fens = []
        
        self._load_all_data()
        print(f"Loaded {len(self.positions)} validation positions")
    
    def _load_all_data(self):
        """Load all validation data into memory."""
        with open(self.data_file, 'rb') as f:
            reader = zstd.ZstdDecompressor().stream_reader(f)
            buffer = ""
            
            # Read data in chunks
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                buffer += chunk.decode('utf-8', errors='ignore')
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            data = json.loads(line)
                            position, evaluation, fen = self._parse_position_data(data)
                            
                            if position is not None and evaluation is not None and fen is not None:
                                self.positions.append(position)
                                self.evaluations.append(evaluation)
                                self.fens.append(fen)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            
            # Process any remaining data in buffer
            if buffer.strip():
                try:
                    data = json.loads(buffer)
                    position, evaluation, fen = self._parse_position_data(data)
                    
                    if position is not None and evaluation is not None and fen is not None:
                        self.positions.append(position)
                        self.evaluations.append(evaluation)
                        self.fens.append(fen)
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
    
    def _parse_position_data(self, data: Dict[str, Any]) -> Tuple[Optional[chess.Board], Optional[float], Optional[str]]:
        """Parse a single position from the JSON data."""
        try:
            # Extract FEN
            fen = data['fen']
            try:
                board = chess.Board(fen)
            except (ValueError, TypeError):
                # Skip malformed FEN strings
                return None, None, None
            
            # Extract evaluation from the first entry in evals
            evals = data['evals']
            if not evals:
                return None, None, None
            
            # Get the first evaluation entry
            first_eval = evals[0]
            pvs = first_eval['pvs']
            if not pvs:
                return None, None, None
            
            # Get the primary variation evaluation
            primary_pv = pvs[0]
            
            # Handle both cp and mate evaluations
            if 'cp' in primary_pv:
                cp_eval = primary_pv['cp']
                # Convert to side-to-move perspective
                if not board.turn:  # Black to move
                    cp_eval = -cp_eval
                return board, cp_eval, fen
            elif 'mate' in primary_pv:
                mate_eval = primary_pv['mate']
                # Convert mate to cp value (2000 - 10 * distance to mate)
                if mate_eval > 0:
                    cp_eval = 2000 - 10 * mate_eval  # Positive mate
                else:
                    cp_eval = -2000 - 10 * mate_eval  # Negative mate
                # Convert to side-to-move perspective
                if not board.turn:  # Black to move
                    cp_eval = -cp_eval
                return board, cp_eval, fen
            else:
                return None, None, None
                
        except (KeyError, ValueError, TypeError):
            return None, None, None
    
    def __len__(self) -> int:
        """Return the number of validation positions."""
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a validation example."""
        if idx >= len(self.positions):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.positions)}")
        
        # Get the position and evaluation
        position = self.positions[idx]
        evaluation = self.evaluations[idx]
        
        # Convert evaluation to win probability
        win_prob = self._cp_to_win_probability(evaluation)
        
        # Extract features
        features = self.feature_extractor._board_to_features(position)
        features = np.transpose(features, (2, 0, 1))  # Convert to (18, 8, 8)
        features = torch.from_numpy(features).float()
        
        # Convert to tensor
        target = torch.tensor(win_prob, dtype=torch.float32)
        
        return features, target
    
    def _cp_to_win_probability(self, cp: float) -> float:
        """Convert centipawn evaluation to win probability using logistic function."""
        cp_clamped = np.clip(cp, -5000, 5000)
        # Scale factor: 400 = moderate curve, 200 = steeper, 600 = gentler
        scale_factor = getattr(self, 'cp_to_prob_scale', 400.0)
        win_prob = 1.0 / (1.0 + np.exp(-cp_clamped / scale_factor))
        return float(win_prob)
    
    def get_random_positions(self, num_positions: int = 10) -> List[Tuple[str, float, torch.Tensor]]:
        """Get random positions for validation logging."""
        if num_positions > len(self.positions):
            num_positions = len(self.positions)
        
        random_indices = random.sample(range(len(self.positions)), num_positions)
        
        results = []
        for idx in random_indices:
            fen = self.fens[idx]
            ground_truth_eval = self.evaluations[idx]
            
            # Get features for this position
            features = self.feature_extractor._board_to_features(self.positions[idx])
            features = np.transpose(features, (2, 0, 1))
            features = torch.from_numpy(features).float()
            
            results.append((fen, ground_truth_eval, features))
        
        return results


def test_data_loader():
    """Test function for the data loader."""
    # Test with compressed sample data
    sample_file = "/Users/rajat/chess-game/trainer/localdata/lichess_db_eval.sample.jsonl.zst"
    
    if os.path.exists(sample_file):
        print("Testing data loader with compressed sample data...")
        
        # Test iterator directly
        print("\n1. Testing iterator directly:")
        iterator = ChessIterator(sample_file, max_positions=10, buffer_size=5)
        for i, (features, target) in enumerate(iterator):
            print(f"  Item {i}: features {features.shape}, target {target.item():.4f}")
            if i >= 4:  # Test first 5 items
                break
        
        # Test dataset
        print("\n2. Testing dataset:")
        dataset = ChessDataset(sample_file, max_positions=20, buffer_size=10)
        for i in range(5):
            try:
                features, target = dataset[i]
                print(f"  Item {i}: features {features.shape}, target {target.item():.4f}")
            except StopIteration:
                print(f"  No more data available after {i} items")
                break
        
        # Test data loader
        print("\n3. Testing batch loading:")
        dataloader = ChessDataLoader.create_dataloader(
            sample_file, 
            batch_size=4, 
            max_positions=20,
            buffer_size=10
        )
        
        for i, (batch_features, batch_targets) in enumerate(dataloader):
            print(f"  Batch {i}: features {batch_features.shape}, targets {batch_targets.shape}")
            if i >= 2:  # Test first few batches
                break
    else:
        print(f"Compressed sample file not found: {sample_file}")


if __name__ == "__main__":
    test_data_loader()
