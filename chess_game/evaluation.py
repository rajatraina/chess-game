"""
Chess Position Evaluation Module

This module provides a clean interface for evaluating chess positions.
It's designed to be easily extensible for ML-based evaluation in the future.

Author: Chess Engine Team
"""

import chess
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os


import numpy as np

# Game stage constants
OPENING = 0
MIDDLEGAME = 1
ENDGAME = 2

# Mapping from piece type to board bitboard attribute for efficient direct access
PIECE_TYPE_TO_BITBOARD = {
    chess.PAWN: lambda b: b.pawns,
    chess.KNIGHT: lambda b: b.knights,
    chess.BISHOP: lambda b: b.bishops,
    chess.ROOK: lambda b: b.rooks,
    chess.QUEEN: lambda b: b.queens,
    chess.KING: lambda b: b.kings,
}


@dataclass
class PieceBitboards:
    """Precomputed bitboards for all piece types and colors for efficient evaluation."""
    white_pawns: int
    black_pawns: int
    white_knights: int
    black_knights: int
    white_bishops: int
    black_bishops: int
    white_rooks: int
    black_rooks: int
    white_queens: int
    black_queens: int
    white_kings: int
    black_kings: int
    castling_flags: int = 0
    piece_count_signature: int = 0
    bitboard_signature: int = 0
    pawn_structure_signature: int = 0
    king_safety_signature: int = 0
    
    def __post_init__(self):
        """Compute signatures from bitboards for caching."""
        # Piece count signature: 40 bits for piece counts (kings excluded)
        self.piece_count_signature = (
            (chess.popcount(self.white_pawns) << 36) |
            (chess.popcount(self.black_pawns) << 32) |
            (chess.popcount(self.white_knights) << 28) |
            (chess.popcount(self.black_knights) << 24) |
            (chess.popcount(self.white_bishops) << 20) |
            (chess.popcount(self.black_bishops) << 16) |
            (chess.popcount(self.white_rooks) << 12) |
            (chess.popcount(self.black_rooks) << 8) |
            (chess.popcount(self.white_queens) << 4) |
            chess.popcount(self.black_queens)
        )
        
        # Bitboard signature: hash of all piece bitboards (used for positional and mobility caches)
        self.bitboard_signature = hash((
            self.white_pawns, self.black_pawns,
            self.white_knights, self.black_knights,
            self.white_bishops, self.black_bishops,
            self.white_rooks, self.black_rooks,
            self.white_queens, self.black_queens,
            self.white_kings, self.black_kings
        ))
        
        # Pawn structure signature: hash of just pawn bitboards
        self.pawn_structure_signature = hash((self.white_pawns, self.black_pawns))
        
        # King safety signature: hash of king bitboards + castling flags + pawn bitboards
        self.king_safety_signature = hash((
            self.white_kings,
            self.black_kings,
            self.castling_flags,
            self.white_pawns,
            self.black_pawns
        ))


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
        
        # Store the starting position's side to move for simplification logic
        self.starting_side_to_move = None
        
        # Evaluation caches: keyed by signatures, cleared at the start of each get_move call
        self.material_cache: Dict[int, float] = {}
        self.positional_cache: Dict[int, float] = {}
        self.mobility_cache: Dict[int, float] = {}
        self.pawn_structure_cache: Dict[int, float] = {}
        self.king_safety_cache: Dict[int, float] = {}
        
    
    def _set_starting_position(self, board: chess.Board):
        """
        Set the starting position for consistent evaluation throughout search.
        
        Args:
            board: The starting position (root position) of the search
        """
        self.starting_piece_count = chess.popcount(board.occupied)
        self.starting_side_to_move = board.turn
    
    def clear_eval_cache(self):
        """Clear all evaluation caches. Called at the start of each get_move."""
        self.material_cache.clear()
        self.positional_cache.clear()
        self.mobility_cache.clear()
        self.pawn_structure_cache.clear()
        self.king_safety_cache.clear()
    
    def _determine_game_stage(self, board: chess.Board) -> int:
        """
        Determine the game stage based on starting position piece count.
        
        Uses starting_piece_count for consistent game stage determination.
        """
        # If there are no queens on the board, treat as endgame
        if not board.queens:
            return ENDGAME
        
        # Check for endgame first
        if self.starting_piece_count is not None:
            if self.starting_piece_count <= 16:
                return ENDGAME
            elif self.starting_piece_count < 32:
                return MIDDLEGAME
            else:  # starting_piece_count == 32
                return OPENING
        else:
            # Fallback: assume middlegame if starting position not set
            return MIDDLEGAME

    def _is_endgame_evaluation(self) -> bool:
        """
        Determine if the current position should be evaluated as an endgame.
        
        Returns:
            True if position is in endgame phase, False otherwise
        """
        if self.starting_piece_count is not None:
            return self.starting_piece_count <= 16
        else:
            return False  # Conservative fallback - assume not endgame if starting position not set

    
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
            "cache_size_limit": 10000,
            "mobility_enabled": {
                "knight": False,
                "bishop": False,
                "rook": False
            },
            "mobility_exclude_pawns_only": True
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
            chess.KNIGHT: mobility_weights.get("2", 3),
            chess.BISHOP: mobility_weights.get("3", 4),
            chess.ROOK: mobility_weights.get("4", 5)
        }
        
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
        
        # Cache pawn structure settings
        self.cached_pawn_structure_enabled = self.config.get("pawn_structure_enabled", True)
        self.cached_pawn_structure_weight = self.config.get("pawn_structure_weight", 0.1)
        self.cached_pawn_islands_bonus = self.config.get("pawn_islands_bonus", 0.1)
        self.cached_passed_pawn_bonus = self.config.get("passed_pawn_bonus", 20)
        
        # Cache game over condition values
        self.cached_draw_value = self.config.get("draw_value", 0)
        self.cached_checkmate_bonus = self.config.get("checkmate_bonus", 100000)
        
        # Cache mobility enabled settings
        mobility_enabled = self.config.get("mobility_enabled", {})
        self.mobility_enabled = {
            chess.KNIGHT: mobility_enabled.get("2", False),
            chess.BISHOP: mobility_enabled.get("3", False),
            chess.ROOK: mobility_enabled.get("4", False)
        }
        
        # Cache mobility exclusion setting (pawns only vs all friendly pieces)
        self.cached_mobility_exclude_pawns_only = self.config.get("mobility_exclude_pawns_only", True)
        
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
            0, 0, 0, 0, 0, 0, 0, 0,     # 1st rank
            0, 0, 0, 0, 0, 0, 0, 0, # 2nd rank - starting position
            4, 4, 8, 15, 15, 8, 4, 4,  # 3rd rank - center control
            2, 2, 6, 12, 12, 6, 2, 2,  # 4th rank - center control
            2, 2, 4, 12, 12, 4, 2, 2,    # 5th rank - center control
            15, 15, 15, 15, 15, 15, 15, 15,    # 6th rank - all good for advancement
            50, 50, 50, 50, 50, 50, 50, 50,    # 7th rank - all good for advancement
            0, 0, 0, 0, 0, 0, 0, 0     # 8th rank
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
            -5, 2, 2, 2, 2, 2, 2, -5,
            -5, 2, 2, 2, 2, 2, 2, -5,
            -5, 2, 2, 2, 2, 2, 2, -5,
            -5, 0, 2, 2, 2, 2, 0, -5,
            -5, 2, 2, 2, 2, 2, 2, -5,
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
            -20, -20, -20, -20, -20, -20, -20, -20, # 1st rank - strongly discourage
            0, 0, 0, 0, 0, 0, 0, 0, # 2nd rank - discourage
            10, 10, 10, 10, 10, 10, 10, 10,        # 3rd rank - neutral
            20, 20, 20, 20, 20, 20, 20, 20,        # 4th rank - moderate value
            40, 40, 40, 40, 40, 40, 40, 40, # 5th rank - good value
            60, 60, 60, 60, 60, 60, 60, 60, # 6th rank - high value
            80, 80, 80, 80, 80, 80, 80, 80, # 7th rank - very high value
            100, 100, 100, 100, 100, 100, 100, 100,      # 8th rank - promotion
        ]
        
        # Endgame king table - encourage centralization
        self.endgame_king_table = [
            -10, -5, 0, 5, 5, 0, -5, -10,   # 1st rank - moderate
            -5, 0, 5, 10, 10, 5, 0, -5,     # 2nd rank - good
            0, 5, 10, 15, 15, 10, 5, 0,      # 3rd rank - very good
            5, 10, 20, 20, 20, 20, 10, 5,    # 4th rank - excellent
            5, 10, 20, 20, 20, 20, 10, 5,    # 5th rank - excellent
            0, 5, 15, 15, 15, 15, 5, 0,      # 6th rank - very good
            -5, 0, 5, 10, 10, 5, 0, -5,     # 7th rank - good
            -10, 0, 0, 5, 5, 0, 0, -10     # 8th rank - moderate
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
    
    def evaluate(self, board: chess.Board, game_stage: int = None) -> float:
        """
        Evaluate the current board position.
        
        Args:
            board: Current chess board state
            game_stage: Game stage (OPENING, MIDDLEGAME, or ENDGAME). If None, will be determined automatically.
            
        Returns:
            Evaluation score from White's perspective
        """
        # Check for game over conditions first
        if board.is_checkmate():
            return self._evaluate_checkmate(board)
        
        if board.is_stalemate() or board.is_insufficient_material():
            return self.cached_draw_value
        
        if board.is_repetition(3):
            return self.cached_draw_value
        
        if board.halfmove_clock >= 100:  # Fifty-move rule
            return self.cached_draw_value
        
        # Determine game stage if not provided
        if game_stage is None:
            game_stage = self._determine_game_stage(board)
        
        # Evaluate position using consolidated function
        return self._evaluate_position(board, game_stage, return_components=False)
    
    def evaluate_with_components(self, board: chess.Board, game_stage: int = None) -> dict:
        """
        Evaluate the current board position with component breakdown.
        
        Args:
            board: Current chess board state
            game_stage: Game stage (OPENING, MIDDLEGAME, or ENDGAME). If None, will be determined automatically.
            
        Returns:
            Dictionary with evaluation components and total score
        """
        # Check for game over conditions first
        if board.is_checkmate():
            checkmate_score = self._evaluate_checkmate(board)
            return {
                'material': 0.0,
                'positional': 0.0,
                'mobility': 0.0,
                'king_safety': 0.0,
                'pawn_structure': 0.0,
                'total': round(checkmate_score, 2)
            }
        
        if board.is_stalemate() or board.is_insufficient_material():
            return {
                'material': 0.0,
                'positional': 0.0,
                'mobility': 0.0,
                'king_safety': 0.0,
                'pawn_structure': 0.0,
                'total': round(self.cached_draw_value, 2)
            }
        
        if board.is_repetition(3):
            return {
                'material': 0.0,
                'positional': 0.0,
                'mobility': 0.0,
                'king_safety': 0.0,
                'pawn_structure': 0.0,
                'total': round(self.cached_draw_value, 2)
            }
        
        if board.halfmove_clock >= 100:  # Fifty-move rule
            return {
                'material': 0.0,
                'positional': 0.0,
                'mobility': 0.0,
                'king_safety': 0.0,
                'pawn_structure': 0.0,
                'total': round(self.cached_draw_value, 2)
            }
        
        # Determine game stage if not provided
        if game_stage is None:
            game_stage = self._determine_game_stage(board)
        
        # Use consolidated function with component breakdown
        return self._evaluate_position(board, game_stage, return_components=True)
    
    def _evaluate_material(self, board: chess.Board, bitboards: PieceBitboards) -> float:
        """Evaluate material balance using optimized bitboard operations with simplification logic"""
        
        # Check cache first using piece_count_signature as key
        signature = bitboards.piece_count_signature
        if signature in self.material_cache:
            return self.material_cache[signature]
        
        # Hoist frequently accessed attributes to locals for performance
        piece_values = self.piece_values
        cached_bishop_pair_bonus = self.cached_bishop_pair_bonus
        
        material_score = 0
        
        # Calculate total material on board for simplification
        total_material_on_board = 0
        
        # Use precomputed bitboards for each piece type
        piece_bitboard_map = {
            chess.PAWN: (bitboards.white_pawns, bitboards.black_pawns),
            chess.KNIGHT: (bitboards.white_knights, bitboards.black_knights),
            chess.BISHOP: (bitboards.white_bishops, bitboards.black_bishops),
            chess.ROOK: (bitboards.white_rooks, bitboards.black_rooks),
            chess.QUEEN: (bitboards.white_queens, bitboards.black_queens),
        }
        
        # Cache bishop counts for reuse
        white_bishops = 0
        black_bishops = 0
        
        for piece_type in piece_values:
            # Use precomputed bitboards
            white_bb, black_bb = piece_bitboard_map[piece_type]
            white_count = chess.popcount(white_bb)
            black_count = chess.popcount(black_bb)
            
            piece_value = piece_values[piece_type]
            material_score += white_count * piece_value
            material_score -= black_count * piece_value
            
            # Track total material on board (both sides)
            total_material_on_board += (white_count + black_count) * piece_value
            
            # Cache bishop counts for bishop pair bonus
            if piece_type == chess.BISHOP:
                white_bishops = white_count
                black_bishops = black_count
        
        # Add bishop pair bonus using cached counts
        if white_bishops >= 2:
            material_score += cached_bishop_pair_bonus
        if black_bishops >= 2:
            material_score -= cached_bishop_pair_bonus
        
        # Material simplification logic for winning positions
        # Apply simplification when the engine's side (starting side to move) is winning
        if self.starting_side_to_move is not None:
            # Check if the engine's side is winning
            engine_is_winning = False
            if self.starting_side_to_move == chess.WHITE and material_score > self.simplification_threshold:
                # Engine is White and White is winning
                engine_is_winning = True
            elif self.starting_side_to_move == chess.BLACK and material_score < -self.simplification_threshold:
                # Engine is Black and Black is winning (negative score means Black is ahead)
                engine_is_winning = True
            
            if engine_is_winning:
                # Calculate simplification factor based on how much material has been traded
                # removed_material is the material that has been captured/traded
                removed_material = self.total_piece_values - total_material_on_board
                
                # Only apply simplification if significant material has been traded
                if removed_material > 1000:  # At least 1000 points of material traded
                    simplification_factor = 1 + self.simplification_multiplier * removed_material
                    
                    # Apply simplification to material score
                    # This makes the engine prefer to maintain its advantage through simplification
                    material_score *= simplification_factor
        
        # Store result in cache before returning
        self.material_cache[signature] = material_score
        return material_score
    
    def _evaluate_positional(self, board: chess.Board, game_stage: int, bitboards: PieceBitboards) -> float:
        """Evaluate positional factors using optimized piece-square tables and mobility"""
        # Check cache first using bitboard_signature as key
        signature = bitboards.bitboard_signature
        if signature in self.positional_cache:
            return self.positional_cache[signature]
        
        # Choose appropriate piece-square tables based on game stage
        tables = self.endgame_piece_square_tables if game_stage == ENDGAME else self.piece_square_tables
        
        positional_score = 0
        
        # Map piece types to precomputed bitboards
        piece_bitboard_map = {
            chess.PAWN: (bitboards.white_pawns, bitboards.black_pawns),
            chess.KNIGHT: (bitboards.white_knights, bitboards.black_knights),
            chess.BISHOP: (bitboards.white_bishops, bitboards.black_bishops),
            chess.ROOK: (bitboards.white_rooks, bitboards.black_rooks),
            chess.QUEEN: (bitboards.white_queens, bitboards.black_queens),
            chess.KING: (bitboards.white_kings, bitboards.black_kings),
        }
        
        for piece_type in tables:
            # Use precomputed bitboards
            white_squares, black_squares = piece_bitboard_map[piece_type]
            
            # Add positional bonuses for White pieces
            for square in chess.scan_forward(white_squares):
                positional_score += tables[piece_type][square]
            
            # Subtract positional bonuses for Black pieces (use cached mirror mapping)
            for square in chess.scan_forward(black_squares):
                positional_score -= tables[piece_type][self.cached_square_mirror[square]]
        
        # Store result in cache before returning
        self.positional_cache[signature] = positional_score
        return positional_score
    
    def _evaluate_mobility(self, board: chess.Board, game_stage: int, bitboards: PieceBitboards) -> float:
        """
        Evaluate piece mobility using efficient bitboard operations.
        
        Mobility is the number of legal moves each piece can make.
        This is a key positional factor that indicates piece activity.
        """
        # Skip mobility evaluation for endgames
        if game_stage == ENDGAME:
            return 0.0
        
        # Check cache first using bitboard_signature as key
        signature = bitboards.bitboard_signature
        if signature in self.mobility_cache:
            return self.mobility_cache[signature]
            
        mobility_score = 0
        
        # Get pieces to exclude during mobility calculation based on configuration
        if self.cached_mobility_exclude_pawns_only:
            # Only exclude pawns (default) - use precomputed bitboards
            white_pieces = bitboards.white_pawns
            black_pieces = bitboards.black_pawns
        else:
            # Exclude all friendly pieces
            white_pieces = board.occupied_co[chess.WHITE]
            black_pieces = board.occupied_co[chess.BLACK]
        
        # Evaluate mobility for each piece type based on configuration
        piece_types = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        
        for piece_type in piece_types:
            # Check if mobility evaluation is enabled for this piece type
            if self.mobility_enabled.get(piece_type, True):  # Default to True if not configured
                mobility_score += self._evaluate_piece_mobility(board, piece_type, chess.WHITE, white_pieces, bitboards)
                mobility_score -= self._evaluate_piece_mobility(board, piece_type, chess.BLACK, black_pieces, bitboards)
        
        # Store result in cache before returning
        self.mobility_cache[signature] = mobility_score
        return mobility_score
    

    def _evaluate_king_safety(self, board: chess.Board, bitboards: PieceBitboards) -> float:
        """
        Evaluate king safety using castling bonus and pawn shield.
        
        Args:
            board: Current board state
            bitboards: Precomputed piece bitboards
            
        Returns:
            King safety score from White's perspective
        """
        # Check cache first using king_safety_signature as key
        signature = bitboards.king_safety_signature
        if signature in self.king_safety_cache:
            return self.king_safety_cache[signature]
        
        king_safety_score = 0
        
        # Evaluate castling bonus
        castling_bonus = self._evaluate_castling_bonus(board)
        king_safety_score += castling_bonus
        
        # Evaluate pawn shield bonus
        pawn_shield_score = self._evaluate_pawn_shield(board, bitboards)
        king_safety_score += pawn_shield_score
        
        # Store result in cache before returning
        self.king_safety_cache[signature] = king_safety_score
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
    
    def _evaluate_pawn_shield(self, board: chess.Board, bitboards: PieceBitboards) -> float:
        """
        Evaluate pawn shield - squares in front of king occupied by friendly pawns.
        
        Args:
            board: Current board state
            bitboards: Precomputed piece bitboards
            
        Returns:
            Pawn shield score from White's perspective
        """
        pawn_shield_score = 0
        
        # Cache king squares to avoid multiple lookups
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # Evaluate White's pawn shield
        if white_king_square is not None:
            white_pawn_shield = self._count_pawn_shield(board, white_king_square, chess.WHITE, bitboards)
            pawn_shield_score += white_pawn_shield * self.cached_pawn_shield_weight
        
        # Evaluate Black's pawn shield
        if black_king_square is not None:
            black_pawn_shield = self._count_pawn_shield(board, black_king_square, chess.BLACK, bitboards)
            pawn_shield_score -= black_pawn_shield * self.cached_pawn_shield_weight
        
        return pawn_shield_score
    
    def _count_pawn_shield(self, board: chess.Board, king_square: int, color: bool, bitboards: PieceBitboards) -> int:
        """
        Count how many shield files have a pawn of our color.
        Only counts when king is on queenside (a-c) or kingside (f-h).
        
        Args:
            board: Current board state
            king_square: Square where the king is located
            color: Color of the pieces (WHITE or BLACK)
            bitboards: Precomputed piece bitboards
            
        Returns:
            1 if all shield files have a pawn of our color, 0 otherwise
        """
        king_file = chess.square_file(king_square)
        
        # Only count pawn shield when king is on queenside or kingside
        if king_file <= 2:  # King is on a, b, c file (queenside)
            # Use files a, b, c (0, 1, 2)
            shield_files = [0, 1, 2]
        elif king_file >= 4:  # King is on e, f, g, or h file (kingside)
            # Use files f, g, h (5, 6, 7)
            shield_files = [5, 6, 7]
        else:  # King is on d file (center) - no pawn shield
            return 0
        
        # Create rank mask for pawn shield ranks (more realistic evaluation)
        if color == chess.WHITE:
            # White pawns: ranks 1-2 (2nd and 3rd ranks, 0-indexed)
            shield_ranks = chess.BB_RANK_2 | chess.BB_RANK_3
        else:
            # Black pawns: ranks 5-6 (6th and 7th ranks, 0-indexed)
            shield_ranks = chess.BB_RANK_6 | chess.BB_RANK_7
        
        # Use precomputed pawn bitboard and apply shield_ranks filter
        our_pawns_bb = bitboards.white_pawns if color == chess.WHITE else bitboards.black_pawns
        our_pawns = our_pawns_bb & shield_ranks
        
        # Check each shield file for pawns on the correct ranks
        # Use boolean array indexed by shield file positions for efficiency
        files_with_pawns = [False] * 3  # shield_files is always length 3
        num_files_with_pawns = 0
        for i, file in enumerate(shield_files):
            file_mask = chess.BB_FILES[file]
            has_pawn = bool(our_pawns & file_mask)
            if has_pawn:
                files_with_pawns[i] = True
                num_files_with_pawns += 1
        
        # New scoring system:
        # - Return 1 if all three shield files have pawns
        # - Return 0.5 if middle file + one other file has pawns
        # - Return 0 otherwise
        if num_files_with_pawns == 3:
            return 1.0  # All shield files covered
        elif files_with_pawns[1] and num_files_with_pawns == 2:
            return 0.5  # Middle file + one other file covered
        return 0.0  # Less than 2 files or middle file not covered
    
    def _evaluate_pawn_structure(self, board: chess.Board, bitboards: PieceBitboards) -> float:
        """
        Evaluate pawn structure using pawn islands and passed pawns.
        
        Pawn islands are calculated by counting contiguous file gaps (files with no pawns).
        The number of pawn islands = number of contiguous file gaps + 1.
        Passed pawns are pawns with no opponent pawns in front on the same or adjacent files.
        
        Args:
            board: Current board state
            bitboards: Precomputed piece bitboards
            
        Returns:
            Pawn structure score from White's perspective (fewer islands is better, more passed pawns is better)
        """
        # Check if pawn structure evaluation is enabled
        if not self.cached_pawn_structure_enabled:
            return 0.0
        
        # Check cache first using pawn_structure_signature as key
        signature = bitboards.pawn_structure_signature
        if signature in self.pawn_structure_cache:
            return self.pawn_structure_cache[signature]
        
        # Use precomputed pawn bitboards
        white_pawns = bitboards.white_pawns
        black_pawns = bitboards.black_pawns
        white_files = [0] * 8  # Files a-h (0-7) - store bitboards
        black_files = [0] * 8
        
        for file in range(8):
            file_mask = chess.BB_FILES[file]
            white_files[file] = white_pawns & file_mask
            black_files[file] = black_pawns & file_mask
        
        # Calculate pawn islands and passed pawns using precomputed file information
        white_islands, white_passed = self._evaluate_pawn_structure_with_files(board, chess.WHITE, white_files, black_files)
        black_islands, black_passed = self._evaluate_pawn_structure_with_files(board, chess.BLACK, black_files, white_files)
        
        # Calculate scores
        # Fewer islands is better, so we want (black_islands - white_islands)
        islands_score = (black_islands - white_islands) * self.cached_pawn_islands_bonus
        
        # More passed pawns is better, so we want (white_passed - black_passed)
        passed_pawns_score = (white_passed - black_passed) * self.cached_passed_pawn_bonus
        
        pawn_structure_score = islands_score + passed_pawns_score
        
        # Store result in cache before returning
        self.pawn_structure_cache[signature] = pawn_structure_score
        return pawn_structure_score
    
    def _evaluate_pawn_structure_with_files(self, board: chess.Board, color: bool, our_files: list[bool], enemy_files: list[bool]) -> tuple[int, float]:
        """
        Analyze pawn structure using precomputed file occupancy information.
        
        Args:
            board: Current board state
            color: Color of the pawns to analyze (WHITE or BLACK)
            our_files: Precomputed file occupancy for our color
            enemy_files: Precomputed file occupancy for enemy color
            
        Returns:
            Tuple of (island_count, passed_pawn_count) where passed_pawn_count can be fractional
        """
        # Count islands using precomputed file information
        islands = 0
        last_file_with_pawns = -1
        has_any_pawns = False
        
        for file in range(8):
            if our_files[file]:
                has_any_pawns = True
                # Check for island gap
                if last_file_with_pawns != -1 and file - last_file_with_pawns > 1:
                    islands += 1
                last_file_with_pawns = file
        
        # Number of islands = number of gaps + 1
        islands = islands + 1 if has_any_pawns else 0
        
        # Count passed pawns using precomputed file information
        passed_pawns = 0
        
        for file in range(8):
            our_pawns_on_file = our_files[file]  # Use precomputed bitboard
            if not our_pawns_on_file:
                continue
                
            # Check if the most advanced pawn on this file is passed
            # Find the most advanced pawn on this file
            most_advanced_rank = -1
            most_advanced_square = None
            
            for square in chess.scan_forward(our_pawns_on_file):
                rank = chess.square_rank(square)
                if color == chess.WHITE:
                    # For white, higher rank is more advanced
                    if rank > most_advanced_rank:
                        most_advanced_rank = rank
                        most_advanced_square = square
                else:
                    # For black, lower rank is more advanced
                    if rank < most_advanced_rank or most_advanced_rank == -1:
                        most_advanced_rank = rank
                        most_advanced_square = square
            
            if most_advanced_square is not None:
                # Skip passed pawn calculation if pawn is too far from promotion
                if color == chess.WHITE:
                    if most_advanced_rank < 4:  # Rank 4 or lower (0-indexed: rank 3 or lower)
                        continue
                else:
                    if most_advanced_rank > 3:  # Rank 5 or higher (0-indexed: rank 4 or higher)
                        continue
                
                # Check if the most advanced pawn is passed
                is_passed = True
                
                # Check adjacent files for enemy pawns
                for check_file in range(max(0, file - 1), min(8, file + 2)):
                    if not enemy_files[check_file]:
                        continue
                        
                    # Check if any enemy pawn on this file is in front of our pawn
                    for enemy_square in chess.scan_forward(enemy_files[check_file]):
                        enemy_rank = chess.square_rank(enemy_square)
                        
                        if color == chess.WHITE:
                            # White pawn: enemy pawn is in front if it's on a higher rank
                            if enemy_rank > most_advanced_rank:
                                is_passed = False
                                break
                        else:
                            # Black pawn: enemy pawn is in front if it's on a lower rank
                            if enemy_rank < most_advanced_rank:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                
                # Note: Multiple pawns same file count as at most one passed pawn
                # Weight passed pawns by how far advanced they are
                if is_passed:
                    if color == chess.WHITE:
                        # White pawns: rank 7 (rank 6 0-indexed) = 2.0, rank 6+ = full value, rank 5 = half value, rank 4- = 0.25
                        if most_advanced_rank == 6:  # Rank 7 (0-indexed: rank 6)
                            passed_pawns += 2.0
                        elif most_advanced_rank >= 5:  # Rank 6 or higher (0-indexed: rank 5+)
                            passed_pawns += 1.0
                        elif most_advanced_rank == 4:  # Rank 5 (0-indexed: rank 4)
                            passed_pawns += 0.5
                        else:  # Rank 3 and below (ranks 4 and below) = 0.25
                            passed_pawns += 0.25
                    else:
                        # Black pawns: rank 2 (rank 1 0-indexed) = 2.0, rank 3- = full value, rank 4 = half value, rank 5+ = 0.25
                        if most_advanced_rank == 1:  # Rank 2 (0-indexed: rank 1)
                            passed_pawns += 2.0
                        elif most_advanced_rank <= 2:  # Rank 3 or lower (0-indexed: rank 2-)
                            passed_pawns += 1.0
                        elif most_advanced_rank == 3:  # Rank 4 (0-indexed: rank 3)
                            passed_pawns += 0.5
                        else:  # Rank 4 and above (ranks 5 and above) = 0.25
                            passed_pawns += 0.25
        
        return islands, passed_pawns
    
    def _evaluate_piece_mobility(self, board: chess.Board, piece_type: int, color: bool, 
                                friendly_pieces: int, bitboards: PieceBitboards) -> float:
        """
        Evaluate mobility for a specific piece type and color using quality-based scoring.
        
        Args:
            board: Current board state
            piece_type: Type of piece to evaluate (KNIGHT, BISHOP, ROOK, QUEEN)
            color: Color of the pieces (WHITE or BLACK)
            friendly_pieces: Bitboard of friendly pieces to exclude (pawns only or all pieces based on config)
            bitboards: Precomputed piece bitboards
            
        Returns:
            Quality-weighted mobility score for this piece type and color
        """
        mobility_score = 0
        
        # Use precomputed bitboards directly based on piece type and color
        if piece_type == chess.KNIGHT:
            piece_squares = bitboards.white_knights if color == chess.WHITE else bitboards.black_knights
        elif piece_type == chess.BISHOP:
            piece_squares = bitboards.white_bishops if color == chess.WHITE else bitboards.black_bishops
        elif piece_type == chess.ROOK:
            piece_squares = bitboards.white_rooks if color == chess.WHITE else bitboards.black_rooks
        else:
            # Should not happen for mobility evaluation, but handle gracefully
            return 0
        
        # Early exit if no pieces of this type
        if not piece_squares:
            return 0
        
        # Use cached values for performance
        weight = self.cached_mobility_weights[piece_type]
        
        # Use precomputed pawn bitboards for rook evaluation
        if piece_type == chess.ROOK:
            white_pawns_bb = bitboards.white_pawns
            black_pawns_bb = bitboards.black_pawns
        
        # Calculate mobility for each piece
        for square in chess.scan_forward(piece_squares):
            if piece_type == chess.KNIGHT:
                # Use knight attack bitboard (integer)
                attacks = chess.BB_KNIGHT_ATTACKS[square]
                legal_moves = attacks & ~friendly_pieces
                
                # Count total number of moves
                mobility_score += chess.popcount(legal_moves) * weight
                
            elif piece_type == chess.BISHOP:
                # Use diagonal attacks with magic bitboard lookup
                diag_attacks = chess.BB_DIAG_ATTACKS[square]
                # Get exact attack bitboard for current occupancy
                diagonal_occupancy = board.occupied & chess.BB_DIAG_MASKS[square]
                if diagonal_occupancy in diag_attacks:
                    attack_bitboard = diag_attacks[diagonal_occupancy]
                    legal_moves = attack_bitboard & ~friendly_pieces
                    
                    # Count total number of moves
                    mobility_score += chess.popcount(legal_moves) * weight
                else:
                    # Fallback to approximation if pattern not found
                    mobility_score += weight * 3  # Default approximation
                    
            elif piece_type == chess.ROOK:
                # Simplified rook mobility: evaluate file control
                file = chess.square_file(square)
                file_mask = chess.BB_FILES[file]
                
                # Check if file is open (no pawns) or half-open (only enemy pawns)
                if color == chess.WHITE:
                    # White rook: check if file is open or half-open for White
                    white_pawns_on_file = white_pawns_bb & file_mask
                    if not white_pawns_on_file:  # No white pawns on file
                        black_pawns_on_file = black_pawns_bb & file_mask
                        if not black_pawns_on_file:
                            # Open file (no pawns)
                            mobility_score += weight * 8  # High bonus for open file
                        else:
                            # Half-open file (only black pawns)
                            mobility_score += weight * 6  # Moderate bonus for half-open file
                else:
                    # Black rook: check if file is open or half-open for Black
                    black_pawns_on_file = black_pawns_bb & file_mask
                    if not black_pawns_on_file:  # No black pawns on file
                        white_pawns_on_file = white_pawns_bb & file_mask
                        if not white_pawns_on_file:
                            # Open file (no pawns)
                            mobility_score += weight * 8  # High bonus for open file
                        else:
                            # Half-open file (only white pawns)
                            mobility_score += weight * 6  # Moderate bonus for half-open file
                    

        
        return mobility_score
    
    
    def _evaluate_position(self, board: chess.Board, game_stage: int, return_components: bool = False):
        """
        Evaluate position with game stage-specific logic.
        
        Args:
            board: Current chess board state
            game_stage: Game stage (OPENING, MIDDLEGAME, or ENDGAME)
            return_components: If True, return dict with component breakdown; if False, return float score
            
        Returns:
            Position evaluation score (float) or component breakdown (dict)
        """
        # Phase 1: Precompute all piece-type × color bitboards once for efficiency
        # Compute castling flags for king safety signature
        castling_flags = (
            ((1 if board.has_kingside_castling_rights(chess.WHITE) else 0) << 0) |
            ((1 if board.has_queenside_castling_rights(chess.WHITE) else 0) << 1) |
            ((1 if board.has_kingside_castling_rights(chess.BLACK) else 0) << 2) |
            ((1 if board.has_queenside_castling_rights(chess.BLACK) else 0) << 3)
        )
        bitboards = PieceBitboards(
            white_pawns=board.pawns & board.occupied_co[chess.WHITE],
            black_pawns=board.pawns & board.occupied_co[chess.BLACK],
            white_knights=board.knights & board.occupied_co[chess.WHITE],
            black_knights=board.knights & board.occupied_co[chess.BLACK],
            white_bishops=board.bishops & board.occupied_co[chess.WHITE],
            black_bishops=board.bishops & board.occupied_co[chess.BLACK],
            white_rooks=board.rooks & board.occupied_co[chess.WHITE],
            black_rooks=board.rooks & board.occupied_co[chess.BLACK],
            white_queens=board.queens & board.occupied_co[chess.WHITE],
            black_queens=board.queens & board.occupied_co[chess.BLACK],
            white_kings=board.kings & board.occupied_co[chess.WHITE],
            black_kings=board.kings & board.occupied_co[chess.BLACK],
            castling_flags=castling_flags,
        )
        
        # Calculate all components using precomputed bitboards
        material_score = self._evaluate_material(board, bitboards)
        positional_score = self._evaluate_positional(board, game_stage, bitboards)
        mobility_score = self._evaluate_mobility(board, game_stage, bitboards)
        pawn_structure_score = self._evaluate_pawn_structure(board, bitboards)
        
        # Choose weights based on game stage
        if game_stage == ENDGAME:
            # Use cached endgame weights for performance
            material_weight = self.cached_endgame_material_weight
            positional_weight = self.cached_endgame_positional_weight
            mobility_weight = self.cached_endgame_mobility_weight
            king_safety_weight = 0.0  # No king safety evaluation in endgame
        else:
            # Use cached standard weights for performance
            material_weight = self.cached_material_weight
            positional_weight = self.cached_positional_weight
            mobility_weight = self.cached_mobility_weight
            king_safety_weight = self.cached_king_safety_weight
        
        # Calculate king safety only for non-endgame positions
        if game_stage != ENDGAME:
            king_safety_score = self._evaluate_king_safety(board, bitboards)
        else:
            king_safety_score = 0.0
        
        # Apply weights
        weighted_material = material_weight * material_score
        weighted_position = positional_weight * positional_score
        weighted_mobility = mobility_weight * mobility_score
        weighted_king_safety = king_safety_weight * king_safety_score
        weighted_pawn_structure = self.cached_pawn_structure_weight * pawn_structure_score
        
        total_score = weighted_material + weighted_position + weighted_mobility + weighted_king_safety + weighted_pawn_structure
        
        if return_components:
            return {
                'material': round(weighted_material, 3),
                'positional': round(weighted_position, 3),
                'mobility': round(weighted_mobility, 3),
                'king_safety': round(weighted_king_safety, 3),
                'pawn_structure': round(weighted_pawn_structure, 3),
                'total': round(total_score, 3)
            }
        else:
            return total_score
    

    
    def _evaluate_checkmate(self, board: chess.Board, distance_to_mate: int = 0) -> float:
        """Evaluate checkmate positions"""
        # When board.is_checkmate() is True, the side to move is checkmated
        # So if board.turn is True, White is checkmated (good for Black)
        # If board.turn is False, Black is checkmated (good for White)
        if board.turn:  # White is checkmated (good for Black)
            return -(self.cached_checkmate_bonus - distance_to_mate)
        else:  # Black is checkmated (good for White)
            return self.cached_checkmate_bonus - distance_to_mate
    
    def get_name(self) -> str:
        return "HandcraftedEvaluator"
    
    def get_description(self) -> str:
        return "Traditional evaluation using material and positional scoring"




# NeuralNetworkEvaluator has been moved to neural_network_evaluator.py
# Import it only when needed to avoid circular imports


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
    
    def _create_evaluator(self, evaluator_type: str, **kwargs) -> BaseEvaluator:
        """Create the specified evaluator"""
        if evaluator_type.lower() == "handcrafted":
            return HandcraftedEvaluator(**kwargs)
        elif evaluator_type.lower() == "neural":
            # Import NeuralNetworkEvaluator only when needed to avoid circular imports
            try:
                from .neural_network_evaluator import NeuralNetworkEvaluator
            except ImportError:
                from neural_network_evaluator import NeuralNetworkEvaluator
            return NeuralNetworkEvaluator(**kwargs)
        else:
            print(f"⚠️  Unknown evaluator type: {evaluator_type}, using handcrafted")
            return HandcraftedEvaluator(**kwargs)
    
    def evaluate(self, board: chess.Board, game_stage: int = None) -> float:
        """
        Evaluate a chess position.
        
        Args:
            board: Current chess board state
            game_stage: Game stage (OPENING, MIDDLEGAME, or ENDGAME). If None, will be determined automatically.
            
        Returns:
            Evaluation score from White's perspective
        """
        score = self.evaluator.evaluate(board, game_stage)
        return score
    
    def evaluate_with_components(self, board: chess.Board, game_stage: int = None) -> dict:
        """
        Evaluate a chess position with component breakdown.
        
        Args:
            board: Current chess board state
            game_stage: Game stage (OPENING, MIDDLEGAME, or ENDGAME). If None, will be determined automatically.
            
        Returns:
            Dictionary with evaluation components and total score
        """
        if hasattr(self.evaluator, 'evaluate_with_components'):
            components = self.evaluator.evaluate_with_components(board, game_stage)
        else:
            # Fallback for evaluators that don't support component breakdown
            score = self.evaluator.evaluate(board, game_stage)
            components = {
                'material': 0.0,
                'positional': 0.0,
                'mobility': 0.0,
                'total': round(score, 2)
            }
        
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
        # Import NeuralNetworkEvaluator only when needed to avoid circular imports
        try:
            from .neural_network_evaluator import NeuralNetworkEvaluator
        except ImportError:
            from neural_network_evaluator import NeuralNetworkEvaluator
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