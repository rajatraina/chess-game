#!/usr/bin/env python3
"""
Evaluation cache system for speeding up repeated position evaluations.
"""

import chess
import time
from typing import Dict, Optional, Tuple

class EvaluationCache:
    """Cache for position evaluations to avoid repeated computation."""
    
    def __init__(self, max_size: int = 100000):
        """
        Initialize evaluation cache.
        
        Args:
            max_size: Maximum number of cached evaluations
        """
        self.cache: Dict[int, Tuple[float, int]] = {}  # hash -> (evaluation, age)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.current_age = 0
    
    def _get_position_hash(self, board: chess.Board) -> int:
        """
        Generate a hash for the current board position.
        Uses a simple but fast hash based on piece positions.
        
        Args:
            board: Current board state
            
        Returns:
            Hash value for the position
        """
        # Simple hash based on piece positions and side to move
        hash_value = 0
        
        # Hash pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Combine piece type, color, and square into hash
                piece_hash = (piece.piece_type << 3) | piece.color
                hash_value ^= (piece_hash << (square * 4))
        
        # Hash side to move
        hash_value ^= (1 << 63) if board.turn else 0
        
        # Hash castling rights
        castling_hash = 0
        if board.has_kingside_castling_rights(chess.WHITE): castling_hash |= 1
        if board.has_queenside_castling_rights(chess.WHITE): castling_hash |= 2
        if board.has_kingside_castling_rights(chess.BLACK): castling_hash |= 4
        if board.has_queenside_castling_rights(chess.BLACK): castling_hash |= 8
        hash_value ^= (castling_hash << 60)
        
        # Hash en passant square
        if board.ep_square is not None:
            hash_value ^= (board.ep_square << 50)
        
        return hash_value
    
    def get(self, board: chess.Board) -> Optional[float]:
        """
        Get cached evaluation for a position.
        
        Args:
            board: Current board state
            
        Returns:
            Cached evaluation if found, None otherwise
        """
        position_hash = self._get_position_hash(board)
        
        if position_hash in self.cache:
            evaluation, _ = self.cache[position_hash]
            self.hits += 1
            return evaluation
        
        self.misses += 1
        return None
    
    def put(self, board: chess.Board, evaluation: float):
        """
        Cache evaluation for a position.
        
        Args:
            board: Current board state
            evaluation: Evaluation score
        """
        position_hash = self._get_position_hash(board)
        
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # Store evaluation with current age
        self.cache[position_hash] = (evaluation, self.current_age)
        self.current_age += 1
    
    def _evict_oldest(self):
        """Evict the oldest cache entry."""
        if not self.cache:
            return
        
        # Find oldest entry
        oldest_hash = min(self.cache.keys(), key=lambda h: self.cache[h][1])
        del self.cache[oldest_hash]
    
    def clear(self):
        """Clear all cached evaluations."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.current_age = 0
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print(f"📊 Evaluation Cache Stats:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.1f}%")
        print(f"  Cache size: {stats['cache_size']}/{stats['max_size']}")

class MaterialOnlyCache:
    """Specialized cache for material-only evaluations."""
    
    def __init__(self, max_size: int = 50000):
        """
        Initialize material-only evaluation cache.
        
        Args:
            max_size: Maximum number of cached evaluations
        """
        self.cache: Dict[int, Tuple[float, int]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.current_age = 0
    
    def _get_material_hash(self, board: chess.Board) -> int:
        """
        Generate a hash for material-only evaluation.
        Only considers piece counts, not positions.
        
        Args:
            board: Current board state
            
        Returns:
            Hash value for material configuration
        """
        # Count pieces for each type and color
        piece_counts = {}
        
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                count = len(board.pieces(piece_type, color))
                piece_counts[(piece_type, color)] = count
        
        # Create hash from piece counts
        hash_value = 0
        for (piece_type, color), count in piece_counts.items():
            # Combine piece type, color, and count
            piece_hash = (piece_type << 3) | color
            hash_value ^= (piece_hash << (count * 4))
        
        return hash_value
    
    def get(self, board: chess.Board) -> Optional[float]:
        """
        Get cached material evaluation.
        
        Args:
            board: Current board state
            
        Returns:
            Cached material evaluation if found, None otherwise
        """
        material_hash = self._get_material_hash(board)
        
        if material_hash in self.cache:
            evaluation, _ = self.cache[material_hash]
            self.hits += 1
            return evaluation
        
        self.misses += 1
        return None
    
    def put(self, board: chess.Board, evaluation: float):
        """
        Cache material evaluation.
        
        Args:
            board: Current board state
            evaluation: Material evaluation score
        """
        material_hash = self._get_material_hash(board)
        
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # Store evaluation with current age
        self.cache[material_hash] = (evaluation, self.current_age)
        self.current_age += 1
    
    def _evict_oldest(self):
        """Evict the oldest cache entry."""
        if not self.cache:
            return
        
        # Find oldest entry
        oldest_hash = min(self.cache.keys(), key=lambda h: self.cache[h][1])
        del self.cache[oldest_hash]
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }
