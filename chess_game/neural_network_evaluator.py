"""
NNUE Chess Position Evaluator

This module provides NNUE (Efficiently Updatable Neural Network) based evaluation 
for chess positions. NNUE is a simple, fast, and interpretable approach to 
chess position evaluation using piece-square table features.

Author: Chess Engine Team
"""

import chess
import os
import torch
import yaml
import json
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import numpy as np

# Import base evaluator from evaluation module
try:
    from .evaluation import BaseEvaluator
except ImportError:
    from evaluation import BaseEvaluator

# Import NNUE model and feature extractor (canonical implementation)
from trainer.nnue_model import NNUE, NNUEFeatureExtractor


class NeuralNetworkEvaluator(BaseEvaluator):
    """
    NNUE-based evaluator for chess position evaluation.
    
    This evaluator uses NNUE (Efficiently Updatable Neural Network) for
    chess position evaluation, which is simple, fast, and interpretable.
    """
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize NNUE evaluator.
        
        Args:
            model_path: Path to trained NNUE model
            config_path: Optional path to NNUE config YAML file (e.g., trainer/config_nnue.yaml)
        """
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use canonical feature extractor from trainer module
        self.feature_extractor = NNUEFeatureExtractor()
        
        # Load engine configuration (for search parameters, not model architecture)
        self.config = self._load_engine_config()
        
        # Initialize attributes for compatibility with engine expectations
        self.starting_piece_count = None
        self.starting_side_to_move = None
        
        self._load_model()
    
    def _load_engine_config(self) -> Dict[str, Any]:
        """
        Load engine configuration from evaluation_config.json.
        This provides search parameters and other engine settings that are
        independent of the evaluation function itself.
        """
        config_file = "chess_game/evaluation_config.json"
        default_config = {
            "search_depth": 4,
            "search_depth_starting": 4,
            "max_search_depth": 16,
            "search_deepening_time_budget_fraction": 0.4,
            "search_deepening_min_time_budget": 6.0,
            "time_budget_check_frequency": 1000,
            "time_budget_early_exit_enabled": True,
            "time_budget_safety_margin": 0.1,
            "time_management_moves_remaining": 35,
            "time_management_moves_remaining_endgame": 20,
            "time_management_increment_used": 0.9,
            "checkmate_bonus": 100000,
            "draw_value": 0,
            "repetition_evaluation": 0,
            "quiescence_additional_depth_limit_shallow_search": 0,
            "quiescence_additional_depth_limit_captures": 8,
            "quiescence_additional_depth_limit_checks": 2,
            "quiescence_additional_depth_limit_promotions": 2,
            "quiescence_additional_depth_limit_queen_defense": 0,
            "quiescence_additional_depth_limit_value_threshold": 0,
            "quiescence_include_checks": True,
            "quiescence_include_queen_defense": False,
            "quiescence_include_value_threshold": False,
            "quiescence_value_threshold": 500,
            "moveorder_shallow_search_depth": 2,
            "search_visualization_enabled": False,
            "search_visualization_target_fen": None,
            "opening_book": {
                "enabled": True,
                "file_path": "opening_book.txt",
                "max_depth": 20,
                "eval_threshold": 0.01,
                "min_games": 1
            },
            "predictive_time_management_enabled": True,
            "predictive_time_safety_factor": 0.8,
            "clear_best_capture_enabled": True,
            "clear_best_capture_threshold": -50,
            "piece_values": {
                "pawn": 100,
                "knight": 330,
                "bishop": 335,
                "rook": 500,
                "queen": 990
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"⚠️  Could not load engine config file {config_file}: {e}")
                print("   Using default engine configuration")
        
        return default_config
    
    def _load_model(self):
        """Load the NNUE model"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                print(f"🤖 Loading NNUE model from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Load model config from YAML file if provided, otherwise use defaults
                model_config = {
                    'hidden_sizes': [256, 32],
                    'dropout': 0.1
                }
                
                if self.config_path and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path, 'r') as f:
                            config = yaml.safe_load(f)
                            if 'model' in config:
                                model_config['hidden_sizes'] = config['model'].get('hidden_sizes', [256, 32])
                                model_config['dropout'] = config['model'].get('dropout', 0.1)
                        print(f"📋 Loaded model architecture from config: {model_config}")
                    except Exception as e:
                        print(f"⚠️  Could not load config file {self.config_path}: {e}")
                        print("   Using default architecture [256, 32]")
                else:
                    print("📋 Using default model architecture [256, 32]")
                
                self.model = NNUE(model_config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print("✅ NNUE model loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load model {self.model_path}: {e}")
        else:
            print("🔧 NNUE feature extractor initialized (no pre-trained model needed for training)")
    
    def _board_to_features(self, board: chess.Board):
        """
        Convert chess board to NNUE piece-square table features.
        
        Uses the canonical NNUEFeatureExtractor from trainer module to ensure
        train/inference consistency.
        
        Args:
            board: Chess board state
            
        Returns:
            Feature vector of shape (782,) for NNUE input
        """
        return self.feature_extractor.board_to_features(board)
    
    
    def _run_neural_network_inference(self, features):
        """
        Run NNUE inference on the feature tensor.
        
        Args:
            features: Feature vector of shape (782,)
            
        Returns:
            Evaluation score from NNUE model
        """
        if self.model is None:
            return None
        
        try:
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                prediction = self.model(features_tensor)
                return float(prediction.cpu().item())
            
        except Exception as e:
            print(f"⚠️  NNUE inference failed: {e}")
            return None
    
    def evaluate(self, board: chess.Board, game_stage: int = None) -> float:
        """
        Evaluate position using NNUE model.
        
        This method only evaluates the position itself, without checking for
        game state conditions like checkmate, stalemate, or repetition.
        Those checks should be handled by the search code.
        
        Args:
            board: Current chess board state
            game_stage: Optional game stage (OPENING, MIDDLEGAME, ENDGAME) - not used by NNUE but kept for compatibility
            
        Returns:
            Evaluation score from White's perspective
        """
        # Check for game over conditions first (same as HandcraftedEvaluator)
        if board.is_checkmate():
            checkmate_bonus = self.config.get("checkmate_bonus", 100000)
            if board.turn:  # White is checkmated (good for Black)
                return -(checkmate_bonus)
            else:  # Black is checkmated (good for White)
                return checkmate_bonus
        
        if board.is_stalemate() or board.is_insufficient_material():
            return self.config.get("draw_value", 0)
        
        if board.is_repetition(3):
            return self.config.get("draw_value", 0)
        
        if board.halfmove_clock >= 100:  # Fifty-move rule
            return self.config.get("draw_value", 0)
        
        # NNUE evaluation
        if self.model is None:
            raise RuntimeError("NNUE model not loaded. Cannot evaluate position.")
        
        try:
            # Convert board to features
            features = self._board_to_features(board)
            if features is None:
                raise RuntimeError("Failed to extract features from board position.")
            
            # Run NNUE inference
            nnue_evaluation = self._run_neural_network_inference(features)
            if nnue_evaluation is None:
                raise RuntimeError("NNUE inference returned None.")
            
            return nnue_evaluation
            
        except Exception as e:
            raise RuntimeError(f"NNUE evaluation failed: {e}")
    
    def _set_starting_position(self, board: chess.Board):
        """
        Set the starting position for consistent evaluation throughout search.
        This is a compatibility method - NNUE doesn't need this but the engine expects it.
        
        Args:
            board: The starting position (root position) of the search
        """
        # NNUE doesn't use starting position for evaluation, but we store it for compatibility
        self.starting_piece_count = chess.popcount(board.occupied)
        self.starting_side_to_move = board.turn
    
    def _is_endgame_evaluation(self) -> bool:
        """
        Determine if the current position should be evaluated as an endgame.
        This is a compatibility method - NNUE doesn't use this but the engine expects it.
        
        Returns:
            True if position is in endgame phase, False otherwise
        """
        if hasattr(self, 'starting_piece_count') and self.starting_piece_count is not None:
            return self.starting_piece_count <= 16
        else:
            return False  # Conservative fallback
    
    def clear_eval_cache(self, game_stage: Optional[int] = None):
        """
        Clear evaluation caches. This is a compatibility method - NNUE doesn't use caches
        but the engine expects this method.
        
        Args:
            game_stage: Current game stage (not used by NNUE)
        """
        # NNUE doesn't use evaluation caches, but we provide this method for compatibility
        pass
    
    def get_name(self) -> str:
        return "NNUEEvaluator"
    
    def get_description(self) -> str:
        return "NNUE-based position evaluation"


class ChessFeatureExtractor:
    """
    Utility class for extracting NNUE features from chess positions.
    
    This class provides methods for converting chess positions
    to NNUE piece-square table features.
    """
    
    @staticmethod
    def board_to_nnue_features(board: chess.Board) -> np.ndarray:
        """
        Convert chess board to NNUE piece-square table features.
        
        Uses the canonical NNUEFeatureExtractor from trainer module to ensure
        train/inference consistency.
        
        Args:
            board: Chess board state
            
        Returns:
            Feature vector of shape (782,)
        """
        extractor = NNUEFeatureExtractor()
        return extractor.board_to_features(board)
    
    @staticmethod
    def board_to_bitboard(board: chess.Board) -> np.ndarray:
        """
        Convert chess board to bitboard representation.
        
        Args:
            board: Chess board state
            
        Returns:
            Array of 12 bitboards (6 piece types × 2 colors)
        """
        
        bitboards = np.zeros(12, dtype=np.uint64)
        
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                      chess.ROOK, chess.QUEEN, chess.KING]
        
        for i, piece_type in enumerate(piece_types):
            # White pieces
            white_bb = board.pieces(piece_type, chess.WHITE)
            bitboards[i * 2] = white_bb
            
            # Black pieces
            black_bb = board.pieces(piece_type, chess.BLACK)
            bitboards[i * 2 + 1] = black_bb
        
        return bitboards
    
    @staticmethod
    def board_to_sparse_features(board: chess.Board) -> dict:
        """
        Convert chess board to sparse feature representation.
        
        Args:
            board: Chess board state
            
        Returns:
            Dictionary with sparse features
        """
        features = {
            'piece_positions': {},
            'side_to_move': 1 if board.turn else 0,
            'castling_rights': {
                'white_kingside': board.has_kingside_castling_rights(chess.WHITE),
                'white_queenside': board.has_queenside_castling_rights(chess.WHITE),
                'black_kingside': board.has_kingside_castling_rights(chess.BLACK),
                'black_queenside': board.has_queenside_castling_rights(chess.BLACK),
            },
            'en_passant_square': board.ep_square,
            'fullmove_number': board.fullmove_number,
            'halfmove_clock': board.halfmove_clock
        }
        
        # Add piece positions
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                color_name = 'white' if color else 'black'
                piece_name = chess.piece_name(piece_type)
                key = f"{color_name}_{piece_name}s"
                features['piece_positions'][key] = list(board.pieces(piece_type, color))
        
        return features


# Factory function for easy NNUE evaluator creation
def create_nnue_evaluator(model_path: Optional[str] = None) -> NeuralNetworkEvaluator:
    """
    Factory function to create NNUE evaluators.
    
    Args:
        model_path: Path to trained NNUE model
        
    Returns:
        Configured NNUE evaluator instance
    """
    return NeuralNetworkEvaluator(model_path=model_path)
