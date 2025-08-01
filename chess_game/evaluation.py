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
        
        # Cache frequently used config values for performance
        self._cache_config_values()
    
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
                "queen": 900,
                "king": 20000
            },
            "checkmate_bonus": 100000,
            "draw_value": 0,
            "quiescence_depth_limit": 10,
            "cache_size_limit": 10000
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
        # Cache mobility weights
        mobility_weights = self.config.get("mobility_weights", {})
        self.cached_mobility_weights = {
            chess.KNIGHT: mobility_weights.get("knight", 3),
            chess.BISHOP: mobility_weights.get("bishop", 4),
            chess.ROOK: mobility_weights.get("rook", 5),
            chess.QUEEN: mobility_weights.get("queen", 2)
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
        
        # Cache central squares bitboard
        self.cached_central_squares = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
    
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
        
        # King table - encourage castling and safety
        self.king_table = [
            -5, -10, -15, -20, -20, -15, -10, -5,   # 8th rank - discourage
            -10, -15, -20, -25, -25, -20, -15, -10, # 7th rank - discourage
            -15, -20, -25, -30, -30, -25, -20, -15, # 6th rank - discourage
            -20, -25, -30, -35, -35, -30, -25, -20, # 5th rank - strongly discourage
            -25, -30, -35, -40, -40, -35, -30, -25, # 4th rank - strongly discourage
            -30, -35, -40, -45, -45, -40, -35, -30, # 3rd rank - strongly discourage
            5, 0, -5, -10, -10, -5, 0, 5,            # 2nd rank - encourage castling squares
            5, 10, 30, -5, 0, -5, 30, 5               # 1st rank - encourage castling squares
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
            0, 0, 0, 0, 0, 0, 0, 0,      # 8th rank - promotion
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
        
        # Check if this is an endgame position
        is_endgame = self._is_endgame_position(board)
        
        if is_endgame:
            return self._evaluate_endgame(board)
        else:
            return self._evaluate_middlegame(board)
    
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
        
        # Check if this is an endgame position
        is_endgame = self._is_endgame_position(board)
        
        if is_endgame:
            # Use cached endgame weights for performance
            material_weight = self.cached_endgame_material_weight
            positional_weight = self.cached_endgame_positional_weight
            mobility_weight = self.cached_endgame_mobility_weight
            
            material_score = self._evaluate_material(board)
            mobility_score = self._evaluate_mobility(board)
            
            # Calculate positional score (excluding mobility)
            positional_score = 0
            tables = self.endgame_piece_square_tables  # Use endgame tables
            for piece_type in tables:
                white_squares = board.pieces(piece_type, chess.WHITE)
                black_squares = board.pieces(piece_type, chess.BLACK)
                
                for square in white_squares:
                    positional_score += tables[piece_type][square]
                
                for square in black_squares:
                    positional_score -= tables[piece_type][chess.square_mirror(square)]
            
            # Apply endgame weights
            weighted_material = material_weight * material_score
            weighted_position = positional_weight * positional_score
            weighted_mobility = mobility_weight * mobility_score
        else:
            # Use standard evaluation
            material_score = self._evaluate_material(board)
            mobility_score = self._evaluate_mobility(board)
            
            # Calculate positional score (excluding mobility)
            positional_score = 0
            tables = self.piece_square_tables  # Use standard tables
            for piece_type in tables:
                white_squares = board.pieces(piece_type, chess.WHITE)
                black_squares = board.pieces(piece_type, chess.BLACK)
                
                for square in white_squares:
                    positional_score += tables[piece_type][square]
                
                for square in black_squares:
                    positional_score -= tables[piece_type][chess.square_mirror(square)]
            
            # Apply standard weights
            weighted_material = self.config["material_weight"] * material_score
            weighted_position = self.config["positional_weight"] * positional_score
            weighted_mobility = self.config["mobility_weight"] * mobility_score
        
        total_score = weighted_material + weighted_position + weighted_mobility
        
        return {
            'material': round(weighted_material, 2),
            'position': round(weighted_position, 2),
            'mobility': round(weighted_mobility, 2),
            'total': round(total_score, 2)
        }
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance using optimized bitboard operations"""
        material_score = 0
        
        for piece_type in self.piece_values:
            # Use bitboard operations instead of len() for better performance
            white_bb = board.pieces(piece_type, chess.WHITE)
            black_bb = board.pieces(piece_type, chess.BLACK)
            white_count = chess.popcount(white_bb)
            black_count = chess.popcount(black_bb)
            material_score += white_count * self.piece_values[piece_type]
            material_score -= black_count * self.piece_values[piece_type]
        
        # Add bishop pair bonus
        bishop_pair_bonus = self.config.get("bishop_pair_bonus", 30)
        white_bishops = chess.popcount(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = chess.popcount(board.pieces(chess.BISHOP, chess.BLACK))
        
        if white_bishops >= 2:
            material_score += bishop_pair_bonus
        if black_bishops >= 2:
            material_score -= bishop_pair_bonus
        
        return material_score
    
    def _evaluate_positional(self, board: chess.Board) -> float:
        """Evaluate positional factors using optimized piece-square tables and mobility"""
        # Check if this is an endgame position
        is_endgame = self._is_endgame_position(board)
        
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
            
            # Subtract positional bonuses for Black pieces (mirror the board)
            for square in black_squares:
                positional_score -= tables[piece_type][chess.square_mirror(square)]
        
        # Add mobility evaluation
        mobility_score = self._evaluate_mobility(board)
        positional_score += mobility_score
        
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
        
        # Evaluate mobility for each piece type
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            mobility_score += self._evaluate_piece_mobility(board, piece_type, chess.WHITE, white_pieces, black_pieces)
            mobility_score -= self._evaluate_piece_mobility(board, piece_type, chess.BLACK, black_pieces, white_pieces)
        
        return mobility_score
    
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
                # Use file and rank attacks
                file_attacks = chess.BB_FILE_ATTACKS[square]
                rank_attacks = chess.BB_RANK_ATTACKS[square]
                if isinstance(file_attacks, dict) or isinstance(rank_attacks, dict):
                    # Handle dictionary format - get exact attack bitboards for current occupancy
                    file_occupancy = board.occupied & chess.BB_FILE_MASKS[square]
                    rank_occupancy = board.occupied & chess.BB_RANK_MASKS[square]
                    
                    file_legal_moves = 0
                    rank_legal_moves = 0
                    
                    if isinstance(file_attacks, dict) and file_occupancy in file_attacks:
                        file_attack_bitboard = file_attacks[file_occupancy]
                        file_legal_moves = file_attack_bitboard & ~friendly_pieces
                    elif not isinstance(file_attacks, dict):
                        file_legal_moves = file_attacks & ~friendly_pieces
                    
                    if isinstance(rank_attacks, dict) and rank_occupancy in rank_attacks:
                        rank_attack_bitboard = rank_attacks[rank_occupancy]
                        rank_legal_moves = rank_attack_bitboard & ~friendly_pieces
                    elif not isinstance(rank_attacks, dict):
                        rank_legal_moves = rank_attacks & ~friendly_pieces
                    
                    total_legal_moves = file_legal_moves | rank_legal_moves
                    
                    # Weight moves by square importance
                    central_moves = total_legal_moves & central_squares
                    regular_moves = total_legal_moves & ~central_squares
                    
                    mobility_score += chess.popcount(central_moves) * weight * central_multiplier
                    mobility_score += chess.popcount(regular_moves) * weight * regular_multiplier
                else:
                    # Handle integer format
                    attacks = file_attacks | rank_attacks
                    legal_moves = attacks & ~friendly_pieces
                    
                    # Weight moves by square importance
                    central_moves = legal_moves & central_squares
                    regular_moves = legal_moves & ~central_squares
                    
                    mobility_score += chess.popcount(central_moves) * weight * central_multiplier
                    mobility_score += chess.popcount(regular_moves) * weight * regular_multiplier
                    
            elif piece_type == chess.QUEEN:
                # Combine diagonal, file, and rank attacks
                diag_attacks = chess.BB_DIAG_ATTACKS[square]
                file_attacks = chess.BB_FILE_ATTACKS[square]
                rank_attacks = chess.BB_RANK_ATTACKS[square]
                if isinstance(diag_attacks, dict) or isinstance(file_attacks, dict) or isinstance(rank_attacks, dict):
                    # Handle dictionary format - get exact attack bitboards for current occupancy
                    diagonal_occupancy = board.occupied & chess.BB_DIAG_MASKS[square]
                    file_occupancy = board.occupied & chess.BB_FILE_MASKS[square]
                    rank_occupancy = board.occupied & chess.BB_RANK_MASKS[square]
                    
                    diag_legal_moves = 0
                    file_legal_moves = 0
                    rank_legal_moves = 0
                    
                    if isinstance(diag_attacks, dict) and diagonal_occupancy in diag_attacks:
                        diag_attack_bitboard = diag_attacks[diagonal_occupancy]
                        diag_legal_moves = diag_attack_bitboard & ~friendly_pieces
                    elif not isinstance(diag_attacks, dict):
                        diag_legal_moves = diag_attacks & ~friendly_pieces
                    
                    if isinstance(file_attacks, dict) and file_occupancy in file_attacks:
                        file_attack_bitboard = file_attacks[file_occupancy]
                        file_legal_moves = file_attack_bitboard & ~friendly_pieces
                    elif not isinstance(file_attacks, dict):
                        file_legal_moves = file_attacks & ~friendly_pieces
                    
                    if isinstance(rank_attacks, dict) and rank_occupancy in rank_attacks:
                        rank_attack_bitboard = rank_attacks[rank_occupancy]
                        rank_legal_moves = rank_attack_bitboard & ~friendly_pieces
                    elif not isinstance(rank_attacks, dict):
                        rank_legal_moves = rank_attacks & ~friendly_pieces
                    
                    total_legal_moves = diag_legal_moves | file_legal_moves | rank_legal_moves
                    
                    # Weight moves by square importance
                    central_moves = total_legal_moves & central_squares
                    regular_moves = total_legal_moves & ~central_squares
                    
                    mobility_score += chess.popcount(central_moves) * weight * central_multiplier
                    mobility_score += chess.popcount(regular_moves) * weight * regular_multiplier
                else:
                    # Handle integer format
                    attacks = diag_attacks | file_attacks | rank_attacks
                    legal_moves = attacks & ~friendly_pieces
                    
                    # Weight moves by square importance
                    central_moves = legal_moves & central_squares
                    regular_moves = legal_moves & ~central_squares
                    
                    mobility_score += chess.popcount(central_moves) * weight * central_multiplier
                    mobility_score += chess.popcount(regular_moves) * weight * regular_multiplier
        
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
        positional_score = self._evaluate_positional(board)
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
        positional_score = self._evaluate_positional(board)
        mobility_score = self._evaluate_mobility(board)
        
        # Use cached standard weights for performance
        total_score = (
            self.cached_material_weight * material_score +
            self.cached_positional_weight * positional_score +
            self.cached_mobility_weight * mobility_score
        )
        
        return total_score
    

    
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