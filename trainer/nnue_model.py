"""
NNUE model for chess position evaluation.

This implementation uses a fixed White-oriented feature frame so the training target,
the model output, and the engine contract all share the same perspective.

Architecture:
- Input: 64 squares x 12 piece types = 768
- Plus 4 castling-right features in White/Black order
- Plus 10 piece-count features in White/Black order
- Plus 1 side-to-move feature
- Total input size: 783 features
- Output: single White-perspective score (centipawns, or scaled cp if training uses cp_target_scale)
"""

import torch
import torch.nn as nn
import numpy as np
import chess
from typing import Dict, Any, Optional, Tuple


class NNUE(nn.Module):
    """
    NNUE (Efficiently Updatable Neural Network) for chess position evaluation.

    Piece-square input; final layer is a linear scalar for centipawn regression
    (target may be scaled by 1/cp_target_scale during training).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configuration parameters
        # hidden_sizes is required - list of sizes for each hidden layer
        if 'hidden_sizes' not in config:
            raise ValueError("'hidden_sizes' is required in config. Provide a list of layer sizes, e.g., [256, 32]")
        
        self.hidden_sizes = config['hidden_sizes']
        if not isinstance(self.hidden_sizes, list) or len(self.hidden_sizes) == 0:
            raise ValueError("'hidden_sizes' must be a non-empty list of integers")
        
        self.num_hidden_layers = len(self.hidden_sizes)
        self.dropout = config.get('dropout', 0.1)
        
        # 768 piece-square features + 4 castling + 10 piece counts + 1 side-to-move.
        self.input_size = 64 * 12 + 4 + 10 + 1
        
        # Build hidden layers
        layers = []
        current_size = self.input_size
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            current_size = hidden_size
        
        # Output layer: White-perspective centipawn-like score (regression; no sigmoid)
        layers.append(nn.Linear(current_size, 1))

        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NNUE network.
        
        Args:
            x: Input tensor of shape (batch_size, 783) in fixed White orientation
            
        Returns:
            Tensor of shape (batch_size, 1): regression target units (cp or cp/cp_target_scale).
        """
        return self.network(x)


class NNUEConfig:
    """Configuration class for NNUE model."""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for the NNUE model."""
        return {
            'hidden_sizes': [256, 32],  # First layer: 256, Second layer: 32
            'dropout': 0.1
        }
    
    @staticmethod
    def get_small_config() -> Dict[str, Any]:
        """Get small configuration for faster training/testing."""
        return {
            'hidden_sizes': [128],  # Single hidden layer with 128 neurons
            'dropout': 0.1
        }
    
    @staticmethod
    def get_large_config() -> Dict[str, Any]:
        """Get large configuration for better performance."""
        return {
            'hidden_sizes': [512, 256, 128],  # Three layers with decreasing sizes
            'dropout': 0.1
        }


class NNUEFeatureExtractor:
    """
    Feature extractor for NNUE model using piece-square tables.
    
    This class converts chess positions to piece-square table features,
    which are much simpler and more interpretable than multi-plane representations.
    """
    
    def __init__(self):
        """Initialize the NNUE feature extractor."""
        self.input_size = 64 * 12 + 4 + 10 + 1
        # Piece type to index mapping (6 piece types × 2 colors = 12 total)
        self.piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                           chess.ROOK, chess.QUEEN, chess.KING]
        self.colors = [chess.WHITE, chess.BLACK]
        
        # Create piece-square table indices
        self.piece_square_indices = {}
        idx = 0
        for piece_type in self.piece_types:
            for color in self.colors:
                self.piece_square_indices[(piece_type, color)] = idx
                idx += 1

        # Piece-count order must stay aligned with existing feature indices.
        self.count_piece_types = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN]
        self.count_piece_index = {
            chess.PAWN: 0,
            chess.ROOK: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.QUEEN: 4,
        }
    
    def board_to_features(self, board: chess.Board) -> np.ndarray:
        """
        Convert chess board to NNUE piece-square table features.
        
        This method creates a 783-dimensional feature vector:
        - 768 features: 64 squares × 12 piece types (piece-square table)
        - 4 features: castling rights (White kingside, White queenside, Black kingside, Black queenside)
        - 10 features: piece counts (White pawns/rooks/knights/bishops/queens, Black same)
        - 1 feature: side to move (1.0 if White to move, 0.0 if Black to move)
        
        Args:
            board: Chess board state
            
        Returns:
            Feature vector of shape (783,) for NNUE input
        """
        # Fixed White-oriented feature layout.
        features = np.zeros(self.input_size, dtype=np.float32)
        
        # Fill piece-square and piece-count features in one pass over piece bitboards.
        for color in self.colors:
            count_base = 772 if color == chess.WHITE else 777
            for piece_type in self.piece_types:
                bitboard = int(board.pieces(piece_type, color))
                piece_idx = self.piece_square_indices[(piece_type, color)]

                # Visit only occupied squares for this piece+color.
                bb = bitboard
                while bb:
                    lsb = bb & -bb
                    square = lsb.bit_length() - 1
                    features[square * 12 + piece_idx] = 1.0
                    bb ^= lsb

                # Reuse this bitboard for piece-count features (except king).
                count_idx = self.count_piece_index.get(piece_type)
                if count_idx is not None:
                    features[count_base + count_idx] = float(chess.popcount(bitboard))

        # Add castling rights features (indices 768-771) via castling bitmask.
        castling_rights = board.castling_rights
        features[768] = 1.0 if castling_rights & chess.BB_H1 else 0.0
        features[769] = 1.0 if castling_rights & chess.BB_A1 else 0.0
        features[770] = 1.0 if castling_rights & chess.BB_H8 else 0.0
        features[771] = 1.0 if castling_rights & chess.BB_A8 else 0.0

        # Side to move feature (index 782)
        features[782] = 1.0 if board.turn == chess.WHITE else 0.0
        
        return features
    
    def get_feature_names(self) -> list:
        """
        Get human-readable feature names for debugging.
        
        Returns:
            List of feature names
        """
        piece_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        color_names = ['white', 'black']
        
        feature_names = []
        for square in range(64):
            rank = square // 8
            file = square % 8
            square_name = chr(ord('a') + file) + str(8 - rank)
            
            for piece_name in piece_names:
                for color_name in color_names:
                    feature_names.append(f"{square_name}_{color_name}_{piece_name}")
        
        feature_names.extend([
            "white_kingside_castling",
            "white_queenside_castling",
            "black_kingside_castling",
            "black_queenside_castling",
            "white_pawn_count",
            "white_rook_count",
            "white_knight_count",
            "white_bishop_count",
            "white_queen_count",
            "black_pawn_count",
            "black_rook_count",
            "black_knight_count",
            "black_bishop_count",
            "black_queen_count",
            "white_to_move",
        ])

        return feature_names


def create_nnue_model(config: Optional[Dict[str, Any]] = None) -> NNUE:
    """
    Factory function to create a NNUE model.
    
    Args:
        config: Model configuration dictionary. If None, uses default config.
        
    Returns:
        Initialized NNUE model
    """
    if config is None:
        config = NNUEConfig.get_default_config()
    
    return NNUE(config)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the NNUE model
    config = NNUEConfig.get_default_config()
    model = create_nnue_model(config)
    
    # Create dummy input (batch_size=2, 783 features)
    dummy_input = torch.randn(2, 783)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"NNUE model created with {count_parameters(model):,} parameters")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Test feature extractor
    import chess
    board = chess.Board()
    extractor = NNUEFeatureExtractor()
    features = extractor.board_to_features(board)
    print(f"Feature vector shape: {features.shape}")
    print(f"Non-zero features: {np.sum(features > 0)}")
