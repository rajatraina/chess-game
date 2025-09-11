"""
Opening book implementation for chess engine using WDL (Win-Draw-Loss) statistics.

Format:
- MOVES: [move sequence] - indicates the position reached by the move sequence
- move W | D | L | WDL - available moves with win/draw/loss counts and WDL score
- Empty lines are ignored

Example:
MOVES: 1. e4 e5
Nf3 40123 | 26789 | 22544 | 0.598
Bc4 22345 | 12345 | 10988 | 0.625
d4 11234 | 6789 | 5433 | 0.622

Important: WDL scores are ALWAYS from White's perspective in the book data.
When Black is to move, the engine uses (1 - WDL) to get the correct score from Black's perspective.

The engine will:
1. Track moves played so far
2. Find matching MOVES: section in the book
3. Adjust WDL scores based on who is to move (White: use as-is, Black: use 1-WDL)
4. Filter moves by WDL threshold (configurable)
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
    Opening book implementation using WDL (Win-Draw-Loss) statistics.
    
    The book stores positions as move sequences and provides moves with their
    historical performance statistics. Move selection is based on WDL scores
    with configurable thresholds.
    """
    
    def __init__(self, book_file_path: Optional[str] = None, random_seed: Optional[int] = None):
        self.book_data: Dict[str, List[Tuple[str, int, int, int, float]]] = {}
        self.wdl_threshold = 0.01  # Moves within 0.01 of best WDL are considered
        self.min_games = 100  # Minimum games required for a move to be considered
        
        # Set random seed based on current time if not provided
        if random_seed is None:
            random_seed = int(time.time() * 1000000)  # Use microseconds for better uniqueness
        random.seed(random_seed)
        
        if book_file_path:
            self.load_book(book_file_path)
    
    def load_book(self, file_path: str) -> None:
        """
        Load opening book from text file with WDL format.
        
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
                    # Parse move with WDL statistics
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
        Parse a move line with WDL statistics.
        
        Args:
            line: The move line to parse
            moves_sequence: The current moves sequence this belongs to
            line_num: Line number for error reporting
            
        Raises:
            OpeningBookError: If format is invalid
        """
        # Format: move W | D | L | WDL
        parts = line.split('|')
        
        if len(parts) != 4:
            raise OpeningBookError(f"Line {line_num}: Expected format 'move W | D | L | WDL', got: {line}")
        
        try:
            # First part contains move and wins: "e4 45623"
            first_part = parts[0].strip().split()
            if len(first_part) != 2:
                raise OpeningBookError(f"Line {line_num}: First part must contain move and wins: {parts[0]}")
            
            move = first_part[0]
            wins = int(first_part[1])
            draws = int(parts[1].strip())
            losses = int(parts[2].strip())
            wdl_score = float(parts[3].strip())
            
            # Calculate WDL score if not provided
            total_games = wins + draws + losses
            if total_games == 0:
                raise OpeningBookError(f"Line {line_num}: Total games cannot be zero")
            
            calculated_wdl = (wins + 0.5 * draws) / total_games
            
            # Use provided WDL score or calculated one
            if wdl_score is not None:
                # Validate WDL score matches calculated value (with small tolerance)
                if abs(calculated_wdl - wdl_score) > 0.001:
                    raise OpeningBookError(f"Line {line_num}: WDL score mismatch. Calculated: {calculated_wdl:.3f}, Provided: {wdl_score}")
                final_wdl = wdl_score
            else:
                final_wdl = calculated_wdl
            
            # Store the move data
            if moves_sequence not in self.book_data:
                self.book_data[moves_sequence] = []
            
            self.book_data[moves_sequence].append((move, wins, draws, losses, final_wdl))
            
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
            List of tuples (move_san, weight) where weight is total games played
        """
        # Convert board to move sequence
        moves_sequence = self._board_to_move_sequence(board)
        
        if moves_sequence not in self.book_data:
            return []
        
        # Convert to engine-compatible format: (move, total_games)
        compatible_moves = []
        for move, wins, draws, losses, wdl in self.book_data[moves_sequence]:
            total_games = wins + draws + losses
            compatible_moves.append((move, total_games))
        
        return compatible_moves
    
    def get_available_moves_detailed(self, board: chess.Board) -> List[Tuple[str, int, int, int, float]]:
        """
        Get available moves with detailed WDL statistics.
        
        Args:
            board: Current board state
            
        Returns:
            List of tuples (move_san, wins, draws, losses, wdl_score)
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
        Get opening move for current position using WDL-based selection.
        WDL scores are always from White's perspective, so for Black moves we use 1-WDL.
        
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
            (move, wins, draws, losses, wdl) 
            for move, wins, draws, losses, wdl in available_moves
            if wins + draws + losses >= self.min_games
        ]
        
        if not qualified_moves:
            return None
        
        # Adjust WDL scores based on who is to move
        # WDL scores in book are always from White's perspective
        if board.turn == chess.WHITE:
            # White to move: use WDL scores as-is
            adjusted_moves = [(move, wins, draws, losses, wdl) for move, wins, draws, losses, wdl in qualified_moves]
        else:
            # Black to move: use 1-WDL (invert the score)
            adjusted_moves = [(move, wins, draws, losses, 1.0 - wdl) for move, wins, draws, losses, wdl in qualified_moves]
        
        # Find best adjusted WDL score
        best_wdl = max(wdl for _, _, _, _, wdl in adjusted_moves)
        
        # Filter moves within WDL threshold of best move
        threshold_moves = [
            (move, wins, draws, losses, wdl)
            for move, wins, draws, losses, wdl in adjusted_moves
            if wdl >= best_wdl - self.wdl_threshold
        ]
        
        if not threshold_moves:
            return None
        
        # Random selection from qualifying moves
        selected_move, _, _, _, _ = random.choice(threshold_moves)
        
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
        
        # Calculate average WDL score
        all_wdl_scores = []
        for moves in self.book_data.values():
            for _, _, _, _, wdl in moves:
                all_wdl_scores.append(wdl)
        
        avg_wdl = sum(all_wdl_scores) / len(all_wdl_scores) if all_wdl_scores else 0.0
        
        return {
            'total_positions': total_positions,
            'total_moves': total_moves,
            'average_wdl_score': round(avg_wdl, 3),
            'wdl_threshold': self.wdl_threshold,
            'min_games_threshold': self.min_games
        }
    
    def set_wdl_threshold(self, threshold: float) -> None:
        """
        Set the WDL threshold for move selection.
        
        Args:
            threshold: Moves within this value of the best WDL will be considered
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("WDL threshold must be between 0.0 and 1.0")
        self.wdl_threshold = threshold
    
    def set_min_games(self, min_games: int) -> None:
        """
        Set the minimum games threshold for move selection.
        
        Args:
            min_games: Minimum number of games required for a move to be considered
        """
        if min_games < 0:
            raise ValueError("Minimum games must be non-negative")
        self.min_games = min_games