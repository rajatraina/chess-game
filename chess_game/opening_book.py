"""
Opening book implementation for chess engine using evaluation-based statistics.

Format:
- MOVES: [move sequence] - indicates the position reached by the move sequence
- move count | avg_eval - available moves with count and average evaluation
- Empty lines are ignored

Example:
MOVES: 1. e4 e5
Nc3 615333 | 0.538
Nf3 5785546 | 0.4
f4 842869 | -0.523

Important: Evaluation scores are ALWAYS from White's perspective in the book data.
When Black is to move, the engine uses (-eval) to get the correct score from Black's perspective.

The engine will:
1. Track moves played so far
2. Find matching MOVES: section in the book
3. Adjust eval scores based on who is to move (White: use as-is, Black: use -eval)
4. Filter moves by eval threshold (configurable)
5. Select randomly from qualifying moves
"""

import chess
import random
import time
from typing import Dict, List, Optional, Tuple, Any


class OpeningBookError(Exception):
    """Exception raised for opening book format errors"""
    pass


class OpeningBook:
    """
    Opening book implementation using evaluation-based statistics.
    
    The book stores positions as move sequences and provides moves with their
    evaluation statistics. Move selection is based on evaluation scores
    with configurable thresholds.
    """
    
    def __init__(self, book_file_path: Optional[str] = None, random_seed: Optional[int] = None):
        self.book_data: Dict[str, List[Tuple[str, int, float]]] = {}
        self.eval_threshold = 0.1  # Moves within 0.1 of best eval are considered
        self.min_games = 10  # Minimum games required for a move to be considered
        
        # Set random seed based on current time if not provided
        if random_seed is None:
            random_seed = int(time.time() * 1000000)  # Use microseconds for better uniqueness
        random.seed(random_seed)
        
        if book_file_path:
            self.load_book(book_file_path)
    
    def load_book(self, file_path: str) -> None:
        """
        Load opening book from text file with eval format.
        
        Args:
            file_path: Path to the opening book file
            
        Raises:
            OpeningBookError: If file format is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise OpeningBookError(f"Opening book file not found: {file_path}")
        except Exception as e:
            raise OpeningBookError(f"Error reading opening book file: {e}")
        
        self.book_data = {}
        current_moves_sequence = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            try:
                if line.startswith('MOVES:'):
                    # Parse MOVES: header
                    current_moves_sequence = self._parse_moves_header(line, line_num)
                elif current_moves_sequence is not None:
                    # Parse move with eval statistics
                    self._parse_move_line(line, current_moves_sequence, line_num)
                else:
                    raise OpeningBookError(f"Line {line_num}: Move data found before MOVES: header")
                    
            except Exception as e:
                raise OpeningBookError(f"Error parsing line {line_num}: {e}")
    
    def _parse_moves_header(self, line: str, line_num: int) -> str:
        """
        Parse MOVES: header to extract the move sequence.
        
        Args:
            line: The MOVES: line to parse
            line_num: Line number for error reporting
            
        Returns:
            The move sequence string
            
        Raises:
            OpeningBookError: If format is invalid
        """
        if not line.startswith('MOVES:'):
            raise OpeningBookError(f"Line {line_num}: Expected MOVES: header, got: {line}")
        
        moves_sequence = line[6:].strip()  # Remove 'MOVES:' prefix
        
        # Validate the move sequence format
        if moves_sequence and not self._validate_move_sequence_format(moves_sequence):
            raise OpeningBookError(f"Line {line_num}: Invalid move sequence format: {moves_sequence}")
        
        return moves_sequence
    
    def _validate_move_sequence_format(self, moves_sequence: str) -> bool:
        """
        Validate that a move sequence follows the expected format.
        
        Args:
            moves_sequence: The move sequence to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not moves_sequence:
            return True  # Empty sequence is valid (starting position)
        
        # Basic validation: should contain move numbers and moves
        # Pattern: "1. e4 e5 2. Nf3 Nc6" or similar
        parts = moves_sequence.split()
        
        # Check for alternating move numbers and moves
        for i, part in enumerate(parts):
            if i % 3 == 0:  # Should be move number (1., 2., etc.)
                if not part.endswith('.'):
                    return False
            elif i % 3 == 1:  # Should be White's move
                if not part or part.startswith('.'):
                    return False
            elif i % 3 == 2:  # Should be Black's move (optional)
                if not part or part.startswith('.'):
                    return False
        
        return True
    
    def _parse_move_line(self, line: str, moves_sequence: str, line_num: int) -> None:
        """
        Parse a move line with eval statistics.
        
        Args:
            line: The move line to parse
            moves_sequence: The current moves sequence this belongs to
            line_num: Line number for error reporting
            
        Raises:
            OpeningBookError: If format is invalid
        """
        # Format: move count | avg_eval
        parts = line.split('|')
        
        if len(parts) != 2:
            raise OpeningBookError(f"Line {line_num}: Expected format 'move count | avg_eval', got: {line}")
        
        try:
            # First part contains move and count: "e4 45623"
            first_part = parts[0].strip().split()
            if len(first_part) != 2:
                raise OpeningBookError(f"Line {line_num}: First part must contain move and count: {parts[0]}")
            
            move = first_part[0]
            count = int(first_part[1])
            avg_eval = float(parts[1].strip())
            
            if count <= 0:
                raise OpeningBookError(f"Line {line_num}: Count must be positive, got: {count}")
            
            # Store the move data
            if moves_sequence not in self.book_data:
                self.book_data[moves_sequence] = []
            
            self.book_data[moves_sequence].append((move, count, avg_eval))
            
        except ValueError as e:
            raise OpeningBookError(f"Line {line_num}: Invalid number format: {e}")
    
    def is_in_book(self, board: chess.Board) -> bool:
        """
        Check if the current position is in the opening book.
        
        Args:
            board: Current board state
            
        Returns:
            True if position is in book, False otherwise
        """
        moves_sequence = self._board_to_move_sequence(board)
        return moves_sequence in self.book_data and len(self.book_data[moves_sequence]) > 0
    
    def get_available_moves(self, board: chess.Board) -> List[Tuple[str, int]]:
        """
        Get available moves for current position from opening book.
        Returns format compatible with engine logging: (move_san, weight)
        
        Args:
            board: Current board state
            
        Returns:
            List of tuples (move_san, weight) where weight is count of games with eval data
        """
        # Convert board to move sequence
        moves_sequence = self._board_to_move_sequence(board)
        
        if moves_sequence not in self.book_data:
            return []
        
        # Convert to engine-compatible format: (move, count)
        compatible_moves = []
        for move, count, avg_eval in self.book_data[moves_sequence]:
            compatible_moves.append((move, count))
        
        return compatible_moves
    
    def get_available_moves_detailed(self, board: chess.Board) -> List[Tuple[str, int, float]]:
        """
        Get available moves with detailed eval statistics.
        
        Args:
            board: Current board state
            
        Returns:
            List of tuples (move_san, count, avg_eval)
        """
        # Convert board to move sequence
        moves_sequence = self._board_to_move_sequence(board)
        
        if moves_sequence not in self.book_data:
            return []
        
        return self.book_data[moves_sequence].copy()
    
    def _board_to_move_sequence(self, board: chess.Board) -> str:
        """
        Convert board position to move sequence string.
        
        Args:
            board: Current board state
            
        Returns:
            Move sequence string
        """
        if len(board.move_stack) == 0:
            return ""  # Starting position
        
        moves = []
        temp_board = chess.Board()
        
        for move in board.move_stack:
            move_san = temp_board.san(move)
            temp_board.push(move)
            moves.append(move_san)
        
        # Format as "1. e4 e5 2. Nf3 Nc6" etc.
        result = []
        for i in range(0, len(moves), 2):
            move_num = (i // 2) + 1
            white_move = moves[i]
            black_move = moves[i + 1] if i + 1 < len(moves) else ""
            
            if black_move:
                result.append(f"{move_num}. {white_move} {black_move}")
            else:
                result.append(f"{move_num}. {white_move}")
        
        return " ".join(result)
    
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get opening move for current position using eval-based selection.
        Eval scores are always from White's perspective, so for Black moves we use -eval.
        
        Args:
            board: Current board state
            
        Returns:
            Chess move if found in book, None otherwise
        """
        available_moves = self.get_available_moves_detailed(board)
        
        if not available_moves:
            return None
        
        # Filter moves by minimum games threshold
        qualified_moves = [
            (move, count, avg_eval) 
            for move, count, avg_eval in available_moves
            if count >= self.min_games
        ]
        
        if not qualified_moves:
            return None
        
        # Adjust eval scores based on who is to move
        # Eval scores in book are always from White's perspective
        if board.turn == chess.WHITE:
            # White to move: use eval scores as-is
            adjusted_moves = [(move, count, avg_eval) for move, count, avg_eval in qualified_moves]
        else:
            # Black to move: use -eval (invert the score)
            adjusted_moves = [(move, count, -avg_eval) for move, count, avg_eval in qualified_moves]
        
        # Find best adjusted eval score
        best_eval = max(eval_score for _, _, eval_score in adjusted_moves)
        
        # Filter moves within eval threshold of best move
        threshold_moves = [
            (move, count, eval_score)
            for move, count, eval_score in adjusted_moves
            if eval_score >= best_eval - self.eval_threshold
        ]
        
        if not threshold_moves:
            return None
        
        # Random selection from qualifying moves
        selected_move, _, _ = random.choice(threshold_moves)
        
        try:
            return board.parse_san(selected_move)
        except Exception:
            return None
    
    def get_book_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded opening book.
        
        Returns:
            Dictionary with book statistics
        """
        total_positions = len(self.book_data)
        total_moves = sum(len(moves) for moves in self.book_data.values())
        
        # Calculate average eval score
        all_eval_scores = []
        for moves in self.book_data.values():
            for _, _, avg_eval in moves:
                all_eval_scores.append(avg_eval)
        
        avg_eval = sum(all_eval_scores) / len(all_eval_scores) if all_eval_scores else 0.0
        
        return {
            'total_positions': total_positions,
            'total_moves': total_moves,
            'average_eval_score': round(avg_eval, 3),
            'eval_threshold': self.eval_threshold,
            'min_games_threshold': self.min_games
        }
    
    def set_eval_threshold(self, threshold: float) -> None:
        """
        Set the eval threshold for move selection.
        
        Args:
            threshold: Moves within this value of the best eval will be considered
        """
        if threshold < 0.0:
            raise ValueError("Eval threshold must be non-negative")
        self.eval_threshold = threshold
    
    def set_min_games(self, min_games: int) -> None:
        """
        Set the minimum games threshold for move selection.
        
        Args:
            min_games: Minimum number of games required for a move to be considered
        """
        if min_games < 0:
            raise ValueError("Minimum games must be non-negative")
        self.min_games = min_games