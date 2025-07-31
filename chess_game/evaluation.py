"""
Chess Position Evaluation Module

This module provides a clean interface for evaluating chess positions.
It's designed to be easily extensible for ML-based evaluation in the future.

Author: Chess Engine Team
"""

import chess
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import json
import os

# Optional numpy import for neural network features
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  NumPy not available. Neural network features will be limited.")


class BaseEvaluator(ABC):
    """Abstract base class for chess position evaluators"""
    
    @abstractmethod
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position.
        
        Args:
            board: Current chess board state
            
        Returns:
            Evaluation score from White's perspective:
            - Positive = good for White
            - Negative = good for Black
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this evaluator"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this evaluator"""
        pass


class HandcraftedEvaluator(BaseEvaluator):
    """
    Traditional handcrafted evaluation function using material and positional scoring.
    
    This evaluator combines:
    - Material evaluation (piece values)
    - Positional evaluation (piece-square tables)
    - Tactical evaluation (checkmate, draws)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the handcrafted evaluator.
        
        Args:
            config_file: Optional path to configuration file with weights
        """
        self.config = self._load_config(config_file)
        self._init_piece_values()
        self._init_piece_square_tables()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load evaluation configuration"""
        default_config = {
            "material_weight": 1.0,
            "positional_weight": 0.1,
            "piece_values": {
                "pawn": 100,
                "knight": 320,
                "bishop": 330,
                "rook": 500,
                "queen": 900,
                "king": 20000
            },
            "checkmate_bonus": 100000,
            "draw_value": 0
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"⚠️  Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _init_piece_values(self):
        """Initialize piece values from config"""
        self.piece_values = {
            chess.PAWN: self.config["piece_values"]["pawn"],
            chess.KNIGHT: self.config["piece_values"]["knight"],
            chess.BISHOP: self.config["piece_values"]["bishop"],
            chess.ROOK: self.config["piece_values"]["rook"],
            chess.QUEEN: self.config["piece_values"]["queen"],
            chess.KING: self.config["piece_values"]["king"]
        }
    
    def _init_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation"""
        # Pawn table - encourage center control and advancement
        self.pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            2, 4, 4, -8, -8, 4, 4, 2,
            2, -2, -4, 0, 0, -4, -2, 2,
            0, 0, 0, 8, 8, 0, 0, 0,
            2, 2, 4, 12, 12, 4, 2, 2,
            4, 4, 8, 15, 15, 8, 4, 4,
            25, 25, 25, 25, 25, 25, 25, 25,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        
        # Knight table - encourage central squares
        self.knight_table = [
            -25, -20, -15, -15, -15, -15, -20, -25,
            -20, -10, 0, 0, 0, 0, -10, -20,
            -15, 0, 5, 8, 8, 5, 0, -15,
            -15, 2, 8, 10, 10, 8, 2, -15,
            -15, 0, 8, 10, 10, 8, 0, -15,
            -15, 2, 5, 8, 8, 5, 2, -15,
            -20, -10, 0, 2, 2, 0, -10, -20,
            -25, -20, -15, -15, -15, -15, -20, -25
        ]
        
        # Bishop table - encourage long diagonals
        self.bishop_table = [
            -10, -5, -5, -5, -5, -5, -5, -10,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 2, 5, 5, 2, 0, -5,
            -5, 2, 2, 5, 5, 2, 2, -5,
            -5, 0, 5, 5, 5, 5, 0, -5,
            -5, 5, 5, 5, 5, 5, 5, -5,
            -5, 2, 0, 0, 0, 0, 2, -5,
            -10, -5, -5, -5, -5, -5, -5, -10
        ]
        
        # Rook table - encourage 7th rank and open files
        self.rook_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            2, 4, 4, 4, 4, 4, 4, 2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            0, 0, 0, 2, 2, 0, 0, 0
        ]
        
        # Queen table - encourage central control
        self.queen_table = [
            -10, -5, -5, -2, -2, -5, -5, -10,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 2, 2, 2, 2, 0, -5,
            -2, 0, 2, 2, 2, 2, 0, -2,
            0, 0, 2, 2, 2, 2, 0, -2,
            -5, 2, 2, 2, 2, 2, 0, -5,
            -5, 0, 2, 0, 0, 0, 0, -5,
            -10, -5, -5, -2, -2, -5, -5, -10
        ]
        
        # King table - encourage safety in middlegame, activity in endgame
        self.king_table = [
            -15, -20, -20, -25, -25, -20, -20, -15,
            -15, -20, -20, -25, -25, -20, -20, -15,
            -15, -20, -20, -25, -25, -20, -20, -15,
            -15, -20, -20, -25, -25, -20, -20, -15,
            -10, -15, -15, -20, -20, -15, -15, -10,
            -5, -10, -10, -10, -10, -10, -10, -5,
            10, 10, 0, 0, 0, 0, 10, 10,
            10, 15, 5, 0, 0, 5, 15, 10
        ]
        
        # Combine all tables
        self.piece_square_tables = {
            chess.PAWN: self.pawn_table,
            chess.KNIGHT: self.knight_table,
            chess.BISHOP: self.bishop_table,
            chess.ROOK: self.rook_table,
            chess.QUEEN: self.queen_table,
            chess.KING: self.king_table
        }
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate the current board position.
        
        Args:
            board: Current chess board state
            
        Returns:
            Evaluation score from White's perspective
        """
        # Check for game over conditions first
        if board.is_checkmate():
            return self._evaluate_checkmate(board)
        
        if board.is_stalemate() or board.is_insufficient_material():
            return self.config["draw_value"]
        
        if board.is_repetition(3):
            return self.config["draw_value"]
        
        if board.halfmove_clock >= 100:  # Fifty-move rule
            return self.config["draw_value"]
        
        # Calculate material and positional scores
        material_score = self._evaluate_material(board)
        positional_score = self._evaluate_positional(board)
        
        # Combine scores with weights
        total_score = (
            self.config["material_weight"] * material_score +
            self.config["positional_weight"] * positional_score
        )
        
        return total_score
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance"""
        material_score = 0
        
        for piece_type in self.piece_values:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            material_score += white_count * self.piece_values[piece_type]
            material_score -= black_count * self.piece_values[piece_type]
        
        return material_score
    
    def _evaluate_positional(self, board: chess.Board) -> float:
        """Evaluate positional factors using piece-square tables"""
        positional_score = 0
        
        for piece_type in self.piece_square_tables:
            # Add positional bonuses for White pieces
            for square in board.pieces(piece_type, chess.WHITE):
                positional_score += self.piece_square_tables[piece_type][square]
            
            # Subtract positional bonuses for Black pieces (mirror the board)
            for square in board.pieces(piece_type, chess.BLACK):
                positional_score -= self.piece_square_tables[piece_type][chess.square_mirror(square)]
        
        return positional_score
    
    def _evaluate_checkmate(self, board: chess.Board) -> float:
        """Evaluate checkmate positions"""
        if board.turn:  # White is checkmated
            return -self.config["checkmate_bonus"]
        else:  # Black is checkmated
            return self.config["checkmate_bonus"]
    
    def get_name(self) -> str:
        return "HandcraftedEvaluator"
    
    def get_description(self) -> str:
        return "Traditional evaluation using material and positional scoring"


class NeuralNetworkEvaluator(BaseEvaluator):
    """
    Neural network-based evaluator for future ML integration.
    
    This is a placeholder for future neural network evaluation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize neural network evaluator.
        
        Args:
            model_path: Path to trained neural network model
        """
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the neural network model"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                # TODO: Implement model loading
                # This is a placeholder for future ML integration
                print(f"🤖 Loading neural network model from {self.model_path}")
                # self.model = load_model(self.model_path)
            except Exception as e:
                print(f"⚠️  Could not load model {self.model_path}: {e}")
        else:
            print("🤖 No neural network model available, using fallback evaluator")
    
    def _board_to_features(self, board: chess.Board):
        """
        Convert chess board to neural network input features.
        
        Args:
            board: Chess board state
            
        Returns:
            Feature vector for neural network
        """
        if not NUMPY_AVAILABLE:
            print("⚠️  NumPy not available for feature conversion")
            return None
        
        # TODO: Implement board to feature conversion
        # This is a placeholder for future implementation
        
        # Example feature representation:
        # - Piece positions (12 planes: 6 piece types × 2 colors)
        # - Side to move (1 plane)
        # - Castling rights (4 planes)
        # - En passant square (1 plane)
        # - Move count (1 plane)
        
        features = np.zeros((8, 8, 19), dtype=np.float32)  # 19 feature planes
        
        # Piece positions (12 planes)
        piece_planes = {
            chess.PAWN: 0,
            chess.KNIGHT: 2,
            chess.BISHOP: 4,
            chess.ROOK: 6,
            chess.QUEEN: 8,
            chess.KING: 10
        }
        
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                plane_idx = piece_planes[piece_type] + (0 if color else 1)
                for square in board.pieces(piece_type, color):
                    rank, file = chess.square_rank(square), chess.square_file(square)
                    features[rank, file, plane_idx] = 1.0
        
        # Side to move (plane 12)
        if board.turn:
            features[:, :, 12] = 1.0
        
        # Castling rights (planes 13-16)
        castling_rights = [chess.BB_A1, chess.BB_H1, chess.BB_A8, chess.BB_H8]
        for i, right in enumerate(castling_rights):
            if board.has_castling_rights(right):
                features[:, :, 13 + i] = 1.0
        
        # En passant square (plane 17)
        if board.ep_square is not None:
            rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            features[rank, file, 17] = 1.0
        
        # Move count (plane 18)
        features[:, :, 18] = board.fullmove_number / 100.0  # Normalize
        
        return features
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using neural network.
        
        Args:
            board: Current chess board state
            
        Returns:
            Evaluation score from White's perspective
        """
        if self.model is None:
            # Fallback to handcrafted evaluation
            fallback = HandcraftedEvaluator()
            return fallback.evaluate(board)
        
        try:
            # Convert board to features
            features = self._board_to_features(board)
            
            # TODO: Run neural network inference
            # prediction = self.model.predict(features.reshape(1, 8, 8, 19))
            # return float(prediction[0])
            
            # Placeholder: return handcrafted evaluation
            fallback = HandcraftedEvaluator()
            return fallback.evaluate(board)
            
        except Exception as e:
            print(f"⚠️  Neural network evaluation failed: {e}")
            # Fallback to handcrafted evaluation
            fallback = HandcraftedEvaluator()
            return fallback.evaluate(board)
    
    def get_name(self) -> str:
        return "NeuralNetworkEvaluator"
    
    def get_description(self) -> str:
        return "Neural network-based position evaluation"


class EvaluationManager:
    """
    Manager class for handling different evaluation methods.
    
    This class provides a unified interface for different evaluators
    and allows easy switching between them.
    """
    
    def __init__(self, evaluator_type: str = "handcrafted", **kwargs):
        """
        Initialize evaluation manager.
        
        Args:
            evaluator_type: Type of evaluator to use ("handcrafted" or "neural")
            **kwargs: Additional arguments for the evaluator
        """
        self.evaluator = self._create_evaluator(evaluator_type, **kwargs)
        self.evaluation_history = []
    
    def _create_evaluator(self, evaluator_type: str, **kwargs) -> BaseEvaluator:
        """Create the specified evaluator"""
        if evaluator_type.lower() == "handcrafted":
            return HandcraftedEvaluator(**kwargs)
        elif evaluator_type.lower() == "neural":
            return NeuralNetworkEvaluator(**kwargs)
        else:
            print(f"⚠️  Unknown evaluator type: {evaluator_type}, using handcrafted")
            return HandcraftedEvaluator(**kwargs)
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position.
        
        Args:
            board: Current chess board state
            
        Returns:
            Evaluation score from White's perspective
        """
        score = self.evaluator.evaluate(board)
        
        # Store evaluation history for analysis
        self.evaluation_history.append({
            'fen': board.fen(),
            'score': score,
            'evaluator': self.evaluator.get_name()
        })
        
        return score
    
    def get_evaluator_info(self) -> Dict[str, str]:
        """Get information about the current evaluator"""
        return {
            'name': self.evaluator.get_name(),
            'description': self.evaluator.get_description()
        }
    
    def switch_evaluator(self, evaluator_type: str, **kwargs):
        """Switch to a different evaluator"""
        self.evaluator = self._create_evaluator(evaluator_type, **kwargs)
        print(f"🔄 Switched to {self.evaluator.get_name()}")
    
    def get_evaluation_history(self) -> List[Dict]:
        """Get evaluation history for analysis"""
        return self.evaluation_history.copy()
    
    def clear_history(self):
        """Clear evaluation history"""
        self.evaluation_history.clear()


# Factory function for easy evaluator creation
def create_evaluator(evaluator_type: str = "handcrafted", **kwargs) -> BaseEvaluator:
    """
    Factory function to create evaluators.
    
    Args:
        evaluator_type: Type of evaluator ("handcrafted" or "neural")
        **kwargs: Arguments for the evaluator
        
    Returns:
        Configured evaluator instance
    """
    if evaluator_type.lower() == "handcrafted":
        return HandcraftedEvaluator(**kwargs)
    elif evaluator_type.lower() == "neural":
        return NeuralNetworkEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


# Default evaluator for backward compatibility
def evaluate_position(board: chess.Board) -> float:
    """
    Default evaluation function for backward compatibility.
    
    Args:
        board: Chess board state
        
    Returns:
        Evaluation score from White's perspective
    """
    evaluator = HandcraftedEvaluator()
    return evaluator.evaluate(board) 