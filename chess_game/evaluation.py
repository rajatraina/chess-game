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

try:
    from .evaluation_cache import EvaluationCache, MaterialOnlyCache
except ImportError:
    from evaluation_cache import EvaluationCache, MaterialOnlyCache

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
        
        # Cache frequently used config values for performance
        self._cache_config_values()
        
        # Store the starting position's piece count for consistent evaluation
        self.starting_piece_count = None
        
        # Initialize evaluation caches based on config
        eval_cache_enabled = self.config.get("evaluation_cache_enabled", True)
        eval_cache_size = self.config.get("evaluation_cache_size", 100000)
        material_cache_enabled = self.config.get("material_cache_enabled", True)
        material_cache_size = self.config.get("material_cache_size", 50000)
        
        self.evaluation_cache = EvaluationCache(max_size=eval_cache_size) if eval_cache_enabled else None
        self.material_cache = MaterialOnlyCache(max_size=material_cache_size) if material_cache_enabled else None
    
    def _set_starting_position(self, board: chess.Board):
        """
        Set the starting position for consistent evaluation throughout search.
        
        Args:
            board: The starting position (root position) of the search
        """
        self.starting_piece_count = chess.popcount(board.occupied)
    
    def _is_endgame_evaluation(self) -> bool:
        """
        Determine if we should use endgame evaluation based on starting position.
        
        Returns:
            True if endgame evaluation should be used
        """
        # Use starting position piece count if available, otherwise fall back to current position
        if self.starting_piece_count is not None:
            return self.starting_piece_count <= 12
        else:
            # Fallback for when starting position hasn't been set
            return True  # Conservative fallback
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load evaluation configuration"""
        # Default config is now minimal - most values should come from JSON file
        default_config = {
            "material_weight": 1.0,
            "positional_weight": 0.1,
            "piece_values": {
                "pawn": 100,
                "knight": 320,
                "bishop": 330,
                "rook": 500,
                "queen": 900
            },
            "total_piece_values": 8180,
            "simplification_material_diff_threshold": 600,
            "simplification_material_diff_multiplier": 0.001,
            "checkmate_bonus": 100000,
            "draw_value": 0,
            "quiescence_depth_limit": 10,
            "cache_size_limit": 10000,
            "mobility_enabled": {
                "knight": False,
                "bishop": False,
                "rook": False
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"⚠️  Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _cache_config_values(self):
        """Cache frequently used config values for performance"""
        # Cache mobility weights (excluding queen - use piece-square tables instead)
        mobility_weights = self.config.get("mobility_weights", {})
        self.cached_mobility_weights = {
            chess.KNIGHT: mobility_weights.get("knight", 3),
            chess.BISHOP: mobility_weights.get("bishop", 4),
            chess.ROOK: mobility_weights.get("rook", 5)
        }
        
        # Cache quality weights
        quality_weights = self.config.get("mobility_quality_weights", {})
        self.cached_central_multiplier = quality_weights.get("central_squares_multiplier", 2.0)
        self.cached_regular_multiplier = quality_weights.get("regular_squares_multiplier", 1.0)
        
        # Cache endgame weights
        endgame_weights = self.config.get("endgame_weights", {})
        self.cached_endgame_material_weight = endgame_weights.get("material_weight", 2.0)
        self.cached_endgame_positional_weight = endgame_weights.get("positional_weight", 1.5)
        self.cached_endgame_mobility_weight = endgame_weights.get("mobility_weight", 0.1)
        
        # Cache standard weights
        self.cached_material_weight = self.config.get("material_weight", 1.0)
        self.cached_positional_weight = self.config.get("positional_weight", 0.1)
        self.cached_mobility_weight = self.config.get("mobility_weight", 0.3)
        self.cached_king_safety_weight = self.config.get("king_safety_weight", 1.0)
        
        # Cache central squares bitboard
        self.cached_central_squares = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
        
        # Cache bishop pair bonus
        self.cached_bishop_pair_bonus = self.config.get("bishop_pair_bonus", 30)
        
        # Cache pawn shield weight
        self.cached_pawn_shield_weight = self.config.get("pawn_shield_weight", 5)
        
        # Cache castling bonuses
        self.cached_kingside_castling_bonus = self.config.get("kingside_castling_bonus", 30)
        self.cached_queenside_castling_bonus = self.config.get("queenside_castling_bonus", 25)
        
        # Cache castling rights penalties
        self.cached_kingside_castling_right_penalty = self.config.get("kingside_castling_right_penalty", -15)
        self.cached_queenside_castling_right_penalty = self.config.get("queenside_castling_right_penalty", -12)
        
        # Cache mobility enabled settings
        mobility_enabled = self.config.get("mobility_enabled", {})
        self.mobility_enabled = {
            "knight": mobility_enabled.get("knight", False),
            "bishop": mobility_enabled.get("bishop", False),
            "rook": mobility_enabled.get("rook", False)
        }
        
        # Cache square mirror mapping for positional evaluation
        self.cached_square_mirror = {}
        for square in range(64):
            self.cached_square_mirror[square] = chess.square_mirror(square)
        

    
    def _init_piece_values(self):
        """Initialize piece values from config"""
        self.piece_values = {
            chess.PAWN: self.config["piece_values"]["pawn"],
            chess.KNIGHT: self.config["piece_values"]["knight"],
            chess.BISHOP: self.config["piece_values"]["bishop"],
            chess.ROOK: self.config["piece_values"]["rook"],
            chess.QUEEN: self.config["piece_values"]["queen"]
        }
        
        # Calculate total piece values for material simplification
        # Use config value if available, otherwise calculate from piece values
        self.total_piece_values = self.config.get("total_piece_values", sum(self.piece_values.values()))
        
        # Load simplification parameters
        self.simplification_threshold = self.config.get("simplification_material_diff_threshold", 600)
        self.simplification_multiplier = self.config.get("simplification_material_diff_multiplier", 0.001)
    
    def _init_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation"""
        # Pawn table - encourage center control and advancement
        self.pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            8, 8, 8, 8, 8, 8, 8, 8,    # 7th rank - all good for advancement
            6, 6, 6, 6, 6, 6, 6, 6,    # 6th rank - all good for advancement
            0, 0, 0, 8, 8, 0, 0, 0,    # 5th rank - center control
            2, 2, 4, 12, 12, 4, 2, 2,  # 4th rank - center control
            4, 4, 8, 15, 15, 8, 4, 4,  # 3rd rank - center control
            25, 25, 25, 25, 25, 25, 25, 25, # 2nd rank - starting position
            0, 0, 0, 0, 0, 0, 0, 0     # 1st rank
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
            0, 0, 0, 2, 2, 0, 0, 0,              # 1st rank - starting position
            -2, 0, 0, 0, 0, 0, 0, -2,            # 2nd rank - discourage
            -2, 0, 0, 0, 0, 0, 0, -2,            # 3rd rank - discourage
            -2, 0, 0, 0, 0, 0, 0, -2,            # 4th rank - discourage
            -2, 0, 0, 0, 0, 0, 0, -2,            # 5th rank - discourage
            -2, 0, 0, 0, 0, 0, 0, -2,            # 6th rank - discourage
            2, 4, 4, 4, 4, 4, 4, 2,               # 7th rank - encourage (attack)
            0, 0, 0, 0, 0, 0, 0, 0                # 8th rank - neutral
        ]
        
        # Queen table - penalize central squares, prefer starting position
        self.queen_table = [
            -5, -5, -5, -5, -5, -5, -5, -5,     # 1st rank - safe starting position (baseline)
            -15, -10, -5, -5, -5, -5, -10, -15,  # 2nd rank - less negative than before
            -25, -20, -15, -10, -10, -15, -20, -25,  # 3rd rank - penalize central squares
            -20, -15, -10, -5, -5, -10, -15, -20,    # 4th rank - penalize central squares
            -15, -10, -5, -5, -5, -5, -10, -15,        # 5th rank - penalize central squares
            -20, -15, -10, -5, -5, -10, -15, -20,    # 6th rank - penalize central squares (was positive)
            -25, -20, -15, -10, -10, -15, -20, -25,  # 7th rank - penalize central squares (was moderate)
            -30, -25, -20, -15, -15, -20, -25, -30   # 8th rank - strongly penalize opponent's back rank
        ]
        
        # King table - encourage castling and safety
        self.king_table = [
            5, 10, 30, -5, 0, -5, 30, 5,              # 1st rank - encourage castling squares
            5, 0, -5, -10, -10, -5, 0, 5,             # 2nd rank - encourage castling squares
            -30, -35, -40, -45, -45, -40, -35, -30,  # 3rd rank - strongly discourage
            -25, -30, -35, -40, -40, -35, -30, -25,  # 4th rank - strongly discourage
            -20, -25, -30, -35, -35, -30, -25, -20,  # 5th rank - strongly discourage
            -15, -20, -25, -30, -30, -25, -20, -15,  # 6th rank - discourage
            -10, -15, -20, -25, -25, -20, -15, -10,  # 7th rank - discourage
            -5, -10, -15, -20, -20, -15, -10, -5     # 8th rank - discourage
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
        
        # Create endgame-specific piece-square tables
        self._init_endgame_piece_square_tables()
    
    def _init_endgame_piece_square_tables(self):
        """Initialize endgame-specific piece-square tables"""
        # Endgame pawn table - strongly encourage advancement
        self.endgame_pawn_table = [
            30, 30, 30, 30, 30, 30, 30, 30,      # 8th rank - promotion
            20, 20, 20, 20, 20, 20, 20, 20, # 7th rank - very high value
            15, 15, 15, 15, 15, 15, 15, 15, # 6th rank - high value
            10, 10, 10, 10, 10, 10, 10, 10, # 5th rank - good value
            5, 5, 5, 5, 5, 5, 5, 5,        # 4th rank - moderate value
            0, 0, 0, 0, 0, 0, 0, 0,        # 3rd rank - neutral
            -5, -5, -5, -5, -5, -5, -5, -5, # 2nd rank - discourage
            -10, -10, -10, -10, -10, -10, -10, -10 # 1st rank - strongly discourage
        ]
        
        # Endgame king table - encourage centralization
        self.endgame_king_table = [
            -10, -5, 0, 5, 5, 0, -5, -10,   # 8th rank - moderate
            -5, 0, 5, 10, 10, 5, 0, -5,     # 7th rank - good
            0, 5, 10, 15, 15, 10, 5, 0,      # 6th rank - very good
            5, 10, 15, 20, 20, 15, 10, 5,    # 5th rank - excellent
            5, 10, 15, 20, 20, 15, 10, 5,    # 4th rank - excellent
            0, 5, 10, 15, 15, 10, 5, 0,      # 3rd rank - very good
            -5, 0, 5, 10, 10, 5, 0, -5,     # 2nd rank - good
            -10, -5, 0, 5, 5, 0, -5, -10     # 1st rank - moderate
        ]
        
        # Combine endgame tables
        self.endgame_piece_square_tables = {
            chess.PAWN: self.endgame_pawn_table,
            chess.KNIGHT: self.knight_table,  # Same as middlegame
            chess.BISHOP: self.bishop_table,  # Same as middlegame
            chess.ROOK: self.rook_table,      # Same as middlegame
            chess.QUEEN: self.queen_table,    # Same as middlegame
            chess.KING: self.endgame_king_table
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
        
        # Try to get cached evaluation
        if self.evaluation_cache is not None:
            cached_eval = self.evaluation_cache.get(board)
            if cached_eval is not None:
                return cached_eval
        
        # Check if this should use endgame evaluation based on starting position
        is_endgame = self._is_endgame_evaluation()
        
        if is_endgame:
            evaluation = self._evaluate_endgame(board)
        else:
            evaluation = self._evaluate_middlegame(board)
        
        # Cache the evaluation
        if self.evaluation_cache is not None:
            self.evaluation_cache.put(board, evaluation)
        
        return evaluation
    
    def evaluate_with_components(self, board: chess.Board) -> dict:
        """
        Evaluate the current board position with component breakdown.
        
        Args:
            board: Current chess board state
            
        Returns:
            Dictionary with evaluation components and total score
        """
        # Check for game over conditions first
        if board.is_checkmate():
            checkmate_score = self._evaluate_checkmate(board)
            return {
                'material': 0.0,
                'position': 0.0,
                'mobility': 0.0,
                'total': round(checkmate_score, 2)
            }
        
        if board.is_stalemate() or board.is_insufficient_material():
            return {
                'material': 0.0,
                'position': 0.0,
                'mobility': 0.0,
                'total': round(self.config["draw_value"], 2)
            }
        
        if board.is_repetition(3):
            return {
                'material': 0.0,
                'position': 0.0,
                'mobility': 0.0,
                'total': round(self.config["draw_value"], 2)
            }
        
        if board.halfmove_clock >= 100:  # Fifty-move rule
            return {
                'material': 0.0,
                'position': 0.0,
                'mobility': 0.0,
                'total': round(self.config["draw_value"], 2)
            }
        
        # Check if this should use endgame evaluation based on starting position
        is_endgame = self._is_endgame_evaluation()
        
        if is_endgame:
            # Use cached endgame weights for performance
            material_weight = self.cached_endgame_material_weight
            positional_weight = self.cached_endgame_positional_weight
            mobility_weight = self.cached_endgame_mobility_weight
            
            material_score = self._evaluate_material(board)
            mobility_score = self._evaluate_mobility_adaptive(board)
            
            # Calculate positional score using _evaluate_positional (which only handles PST)
            positional_score = self._evaluate_positional(board, is_endgame=True)
            
            # Apply endgame weights
            weighted_material = material_weight * material_score
            weighted_position = positional_weight * positional_score
            weighted_mobility = mobility_weight * mobility_score
            
            # No king safety evaluation in endgame
            weighted_king_safety = 0.0
        else:
            # Use standard evaluation
            material_score = self._evaluate_material(board)
            mobility_score = self._evaluate_mobility_adaptive(board)
            
            # Calculate positional score using _evaluate_positional (which only handles PST)
            positional_score = self._evaluate_positional(board, is_endgame=False)
            
            # Apply standard weights using cached values for performance
            weighted_material = self.cached_material_weight * material_score
            weighted_position = self.cached_positional_weight * positional_score
            weighted_mobility = self.cached_mobility_weight * mobility_score
            
            # Add king safety evaluation for middlegame
            king_safety_score = self._evaluate_king_safety(board)
            weighted_king_safety = self.cached_king_safety_weight * king_safety_score
        
        total_score = weighted_material + weighted_position + weighted_mobility + weighted_king_safety
        
        return {
            'material': round(weighted_material, 2),
            'position': round(weighted_position, 2),
            'mobility': round(weighted_mobility, 2),
            'king_safety': round(weighted_king_safety, 2),
            'total': round(total_score, 2)
        }
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance using optimized bitboard operations with simplification logic"""
        # Try to get cached material evaluation
        if self.material_cache is not None:
            cached_material = self.material_cache.get(board)
            if cached_material is not None:
                return cached_material
        
        material_score = 0
        
        # Cache bishop counts for reuse
        white_bishops = 0
        black_bishops = 0
        
        # Calculate total material on board for simplification
        total_material_on_board = 0
        
        for piece_type in self.piece_values:
            # Use bitboard operations instead of len() for better performance
            white_bb = board.pieces(piece_type, chess.WHITE)
            black_bb = board.pieces(piece_type, chess.BLACK)
            white_count = chess.popcount(white_bb)
            black_count = chess.popcount(black_bb)
            
            piece_value = self.piece_values[piece_type]
            material_score += white_count * piece_value
            material_score -= black_count * piece_value
            
            # Track total material on board (both sides)
            total_material_on_board += (white_count + black_count) * piece_value
            
            # Cache bishop counts for bishop pair bonus
            if piece_type == chess.BISHOP:
                white_bishops = white_count
                black_bishops = black_count
        
        # Add bishop pair bonus using cached counts
        bishop_pair_bonus = self.cached_bishop_pair_bonus
        if white_bishops >= 2:
            material_score += bishop_pair_bonus
        if black_bishops >= 2:
            material_score -= bishop_pair_bonus
        
        # Apply material simplification logic
        material_diff_abs = abs(material_score)
        if material_diff_abs > self.simplification_threshold:
            # Calculate simplification factor
            # remaining_material is the material that has been captured/traded
            remaining_material = self.total_piece_values - total_material_on_board
            simplification_factor = 1 + self.simplification_multiplier * remaining_material
            
            # Apply simplification to material score
            material_score *= simplification_factor
        
        # Cache the material evaluation
        if self.material_cache is not None:
            self.material_cache.put(board, material_score)
        
        return material_score
    
    def _evaluate_positional(self, board: chess.Board, is_endgame: bool = None) -> float:
        """Evaluate positional factors using optimized piece-square tables and mobility"""
        # Check if this should use endgame evaluation based on starting position (only if not provided)
        if is_endgame is None:
            is_endgame = self._is_endgame_evaluation()
        
        # Choose appropriate piece-square tables
        tables = self.endgame_piece_square_tables if is_endgame else self.piece_square_tables
        
        positional_score = 0
        
        for piece_type in tables:
            # Use bitboard iteration for better performance
            white_squares = board.pieces(piece_type, chess.WHITE)
            black_squares = board.pieces(piece_type, chess.BLACK)
            
            # Add positional bonuses for White pieces
            for square in white_squares:
                positional_score += tables[piece_type][square]
            
            # Subtract positional bonuses for Black pieces (use cached mirror mapping)
            for square in black_squares:
                positional_score -= tables[piece_type][self.cached_square_mirror[square]]
        
        # Note: Mobility and king safety are handled separately in evaluate_with_components
        # to avoid double-counting. This method only handles piece-square tables.
        return positional_score
    
    def _evaluate_mobility(self, board: chess.Board) -> float:
        """
        Evaluate piece mobility using efficient bitboard operations.
        
        Mobility is the number of legal moves each piece can make.
        This is a key positional factor that indicates piece activity.
        """
        mobility_score = 0
        
        # Get all pieces for each side
        white_pieces = board.occupied_co[chess.WHITE]
        black_pieces = board.occupied_co[chess.BLACK]
        
        # Evaluate mobility for each piece type based on configuration
        piece_types = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        piece_names = {chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook"}
        
        for piece_type in piece_types:
            piece_name = piece_names[piece_type]
            
            # Check if mobility evaluation is enabled for this piece type
            if self.mobility_enabled.get(piece_name, True):  # Default to True if not configured
                mobility_score += self._evaluate_piece_mobility(board, piece_type, chess.WHITE, white_pieces, black_pieces)
                mobility_score -= self._evaluate_piece_mobility(board, piece_type, chess.BLACK, black_pieces, white_pieces)
        
        return mobility_score
    

    
    def _evaluate_mobility_adaptive(self, board: chess.Board) -> float:
        """
        Adaptive mobility evaluation that skips mobility for endgames.
        
        Mobility is less important in endgames, so we can skip it entirely.
        """
        # Skip mobility evaluation for endgames
        if self._is_endgame_evaluation():
            return 0.0
        
        # Use full evaluation for middlegame positions
        return self._evaluate_mobility(board)
    
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """
        Evaluate king safety using castling bonus and pawn shield.
        
        Args:
            board: Current board state
            
        Returns:
            King safety score from White's perspective
        """
        king_safety_score = 0
        
        # Evaluate castling bonus
        castling_bonus = self._evaluate_castling_bonus(board)
        king_safety_score += castling_bonus
        
        return king_safety_score
    
    def _evaluate_castling_bonus(self, board: chess.Board) -> float:
        """
        Evaluate castling bonus and castling rights penalties for both sides.
        
        Args:
            board: Current board state
            
        Returns:
            Castling score from White's perspective (bonus for castling, penalty for lost rights)
        """
        castling_score = 0
        
        # Cache king squares to avoid multiple lookups
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # Check if White has castled
        if white_king_square is not None:
            # Kingside castling (g1)
            if white_king_square == chess.G1:
                castling_score += self.cached_kingside_castling_bonus
            # Queenside castling (c1)
            elif white_king_square == chess.C1:
                castling_score += self.cached_queenside_castling_bonus
        
        # Check if Black has castled
        if black_king_square is not None:
            # Kingside castling (g8)
            if black_king_square == chess.G8:
                castling_score -= self.cached_kingside_castling_bonus
            # Queenside castling (c8)
            elif black_king_square == chess.C8:
                castling_score -= self.cached_queenside_castling_bonus
        
        # Evaluate castling rights penalties
        castling_score += self._evaluate_castling_rights_penalties(board)
        
        return castling_score
    
    def _evaluate_castling_rights_penalties(self, board: chess.Board) -> float:
        """
        Evaluate penalties for lost castling rights.
        
        Args:
            board: Current board state
            
        Returns:
            Castling rights penalty score from White's perspective
        """
        castling_rights_penalty = 0
        
        # Check White's castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            # White still has kingside castling rights
            pass
        else:
            # White lost kingside castling rights - apply penalty
            castling_rights_penalty += self.cached_kingside_castling_right_penalty
            
        if board.has_queenside_castling_rights(chess.WHITE):
            # White still has queenside castling rights
            pass
        else:
            # White lost queenside castling rights - apply penalty
            castling_rights_penalty += self.cached_queenside_castling_right_penalty
        
        # Check Black's castling rights
        if board.has_kingside_castling_rights(chess.BLACK):
            # Black still has kingside castling rights
            pass
        else:
            # Black lost kingside castling rights - apply penalty (negative for Black)
            castling_rights_penalty -= self.cached_kingside_castling_right_penalty
            
        if board.has_queenside_castling_rights(chess.BLACK):
            # Black still has queenside castling rights
            pass
        else:
            # Black lost queenside castling rights - apply penalty (negative for Black)
            castling_rights_penalty -= self.cached_queenside_castling_right_penalty
        
        return castling_rights_penalty
    
    def _evaluate_pawn_shield(self, board: chess.Board) -> float:
        """
        Evaluate pawn shield - squares in front of king occupied by friendly pawns.
        
        Args:
            board: Current board state
            
        Returns:
            Pawn shield score from White's perspective
        """
        pawn_shield_score = 0
        
        # Cache king squares to avoid multiple lookups
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # Evaluate White's pawn shield
        if white_king_square is not None:
            white_pawn_shield = self._count_pawn_shield(board, white_king_square, chess.WHITE)
            pawn_shield_score += white_pawn_shield * self.cached_pawn_shield_weight
        
        # Evaluate Black's pawn shield
        if black_king_square is not None:
            black_pawn_shield = self._count_pawn_shield(board, black_king_square, chess.BLACK)
            pawn_shield_score -= black_pawn_shield * self.cached_pawn_shield_weight
        
        return pawn_shield_score
    
    def _count_pawn_shield(self, board: chess.Board, king_square: int, color: bool) -> int:
        """
        Count friendly pawns in the 3 files closest to the king (a-c or f-h).
        
        Args:
            board: Current board state
            king_square: Square where the king is located
            color: Color of the king (WHITE or BLACK)
            
        Returns:
            Number of friendly pawns in the 3 files closest to the king
        """
        king_file = chess.square_file(king_square)
        
        # Determine which 3 files are closest to the king
        if king_file <= 2:  # King is on a, b, or c file (queenside)
            # Use files a, b, c (0, 1, 2)
            shield_files = [0, 1, 2]
        elif king_file >= 5:  # King is on f, g, or h file (kingside)
            # Use files f, g, h (5, 6, 7)
            shield_files = [5, 6, 7]
        else:  # King is on d or e file (center)
            # Use the 3 files closest to the king
            if king_file == 3:  # d file
                shield_files = [2, 3, 4]  # c, d, e
            else:  # e file
                shield_files = [3, 4, 5]  # d, e, f
        
        # Count pawns in the 3 closest files
        pawn_count = 0
        for file in shield_files:
            # Check all ranks for pawns in this file
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_count += 1
        
        return pawn_count
    
    def _evaluate_piece_mobility(self, board: chess.Board, piece_type: int, color: bool, 
                                friendly_pieces: int, enemy_pieces: int) -> float:
        """
        Evaluate mobility for a specific piece type and color using quality-based scoring.
        
        Args:
            board: Current board state
            piece_type: Type of piece to evaluate (KNIGHT, BISHOP, ROOK, QUEEN)
            color: Color of the pieces (WHITE or BLACK)
            friendly_pieces: Bitboard of friendly pieces
            enemy_pieces: Bitboard of enemy pieces
            
        Returns:
            Quality-weighted mobility score for this piece type and color
        """
        mobility_score = 0
        piece_squares = board.pieces(piece_type, color)
        
        # Early exit if no pieces of this type
        if not piece_squares:
            return 0
        
        # Use cached values for performance
        weight = self.cached_mobility_weights[piece_type]
        central_multiplier = self.cached_central_multiplier
        regular_multiplier = self.cached_regular_multiplier
        
        # Use cached central squares bitboard
        central_squares = self.cached_central_squares
        
        # Use quality-based mobility calculation
        for square in piece_squares:
            if piece_type == chess.KNIGHT:
                # Use knight attack bitboard (integer)
                attacks = chess.BB_KNIGHT_ATTACKS[square]
                legal_moves = attacks & ~friendly_pieces
                
                # Weight moves by square importance
                central_moves = legal_moves & central_squares
                regular_moves = legal_moves & ~central_squares
                
                mobility_score += chess.popcount(central_moves) * weight * central_multiplier
                mobility_score += chess.popcount(regular_moves) * weight * regular_multiplier
                
            elif piece_type == chess.BISHOP:
                # Use diagonal attacks with proper handling
                diag_attacks = chess.BB_DIAG_ATTACKS[square]
                if isinstance(diag_attacks, dict):
                    # Handle dictionary format - get exact attack bitboard for current occupancy
                    diagonal_occupancy = board.occupied & chess.BB_DIAG_MASKS[square]
                    if diagonal_occupancy in diag_attacks:
                        attack_bitboard = diag_attacks[diagonal_occupancy]
                        legal_moves = attack_bitboard & ~friendly_pieces
                        
                        # Weight moves by square importance
                        central_moves = legal_moves & central_squares
                        regular_moves = legal_moves & ~central_squares
                        
                        mobility_score += chess.popcount(central_moves) * weight * central_multiplier
                        mobility_score += chess.popcount(regular_moves) * weight * regular_multiplier
                    else:
                        # Fallback to approximation if pattern not found
                        mobility_score += weight * 3  # Default approximation
                else:
                    # Handle integer format
                    legal_moves = diag_attacks & ~friendly_pieces
                    
                    # Weight moves by square importance
                    central_moves = legal_moves & central_squares
                    regular_moves = legal_moves & ~central_squares
                    
                    mobility_score += chess.popcount(central_moves) * weight * central_multiplier
                    mobility_score += chess.popcount(regular_moves) * weight * regular_multiplier
                    
            elif piece_type == chess.ROOK:
                # Simplified rook mobility: evaluate file control
                file = chess.square_file(square)
                file_mask = chess.BB_FILES[file]
                
                # Check if file is open (no pawns) or half-open (only enemy pawns)
                if color == chess.WHITE:
                    # White rook: check if file is open or half-open for White
                    white_pawns_on_file = board.pieces(chess.PAWN, chess.WHITE) & file_mask
                    if not white_pawns_on_file:  # No white pawns on file
                        black_pawns_on_file = board.pieces(chess.PAWN, chess.BLACK) & file_mask
                        if not black_pawns_on_file:
                            # Open file (no pawns)
                            mobility_score += weight * 8  # High bonus for open file
                        else:
                            # Half-open file (only black pawns)
                            mobility_score += weight * 4  # Moderate bonus for half-open file
                else:
                    # Black rook: check if file is open or half-open for Black
                    black_pawns_on_file = board.pieces(chess.PAWN, chess.BLACK) & file_mask
                    if not black_pawns_on_file:  # No black pawns on file
                        white_pawns_on_file = board.pieces(chess.PAWN, chess.WHITE) & file_mask
                        if not white_pawns_on_file:
                            # Open file (no pawns)
                            mobility_score += weight * 8  # High bonus for open file
                        else:
                            # Half-open file (only white pawns)
                            mobility_score += weight * 4  # Moderate bonus for half-open file
                    

        
        return mobility_score
    
    def _is_endgame_position(self, board: chess.Board) -> bool:
        """
        Determine if the position is an endgame.
        
        Args:
            board: Current chess board state
            
        Returns:
            True if this is an endgame position
        """
        # Count total pieces
        total_pieces = chess.popcount(board.occupied)
        
        # Consider it endgame if 12 or fewer pieces
        return total_pieces <= 12
    
    def _evaluate_endgame(self, board: chess.Board) -> float:
        """
        Evaluate endgame positions with endgame-specific logic.
        
        Args:
            board: Current chess board state
            
        Returns:
            Endgame evaluation score
        """
        # Use cached endgame weights for performance
        material_weight = self.cached_endgame_material_weight
        positional_weight = self.cached_endgame_positional_weight
        mobility_weight = self.cached_endgame_mobility_weight
        
        # Calculate components
        material_score = self._evaluate_material(board)
        positional_score = self._evaluate_positional(board, is_endgame=True)
        mobility_score = self._evaluate_mobility(board)
        
        # Apply endgame-specific weights
        total_score = (
            material_weight * material_score +
            positional_weight * positional_score +
            mobility_weight * mobility_score
        )
        
        return total_score
    
    def _evaluate_middlegame(self, board: chess.Board) -> float:
        """
        Evaluate middlegame positions with standard weights.
        
        Args:
            board: Current chess board state
            
        Returns:
            Middlegame evaluation score
        """
        # Calculate all components
        material_score = self._evaluate_material(board)
        positional_score = self._evaluate_positional(board, is_endgame=False)
        mobility_score = self._evaluate_mobility(board)
        
        # Use cached standard weights for performance
        total_score = (
            self.cached_material_weight * material_score +
            self.cached_positional_weight * positional_score +
            self.cached_mobility_weight * mobility_score
        )
        
        return total_score
    

    
    def _evaluate_checkmate(self, board: chess.Board, distance_to_mate: int = 0) -> float:
        """Evaluate checkmate positions"""
        if board.turn:  # White is checkmated
            return -(self.config["checkmate_bonus"] - distance_to_mate)
        else:  # Black is checkmated
            return self.config["checkmate_bonus"] - distance_to_mate
    
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
    
    def evaluate_with_components(self, board: chess.Board) -> dict:
        """
        Evaluate a chess position with component breakdown.
        
        Args:
            board: Current chess board state
            
        Returns:
            Dictionary with evaluation components and total score
        """
        if hasattr(self.evaluator, 'evaluate_with_components'):
            components = self.evaluator.evaluate_with_components(board)
        else:
            # Fallback for evaluators that don't support component breakdown
            score = self.evaluator.evaluate(board)
            components = {
                'material': 0.0,
                'position': 0.0,
                'mobility': 0.0,
                'total': round(score, 2)
            }
        
        # Store evaluation history for analysis
        self.evaluation_history.append({
            'fen': board.fen(),
            'score': components['total'],
            'evaluator': self.evaluator.get_name(),
            'components': components
        })
        
        return components
    
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