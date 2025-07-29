import chess
from chess_game.engine import MinimaxEngine

# Create a test position with a capture
board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
# This is a position where Black can capture the pawn on e4

engine = MinimaxEngine(depth=2)

# Test a specific capture move
move = chess.Move.from_uci("d7e4")  # Black pawn captures White pawn on e4
engine.test_capture_evaluation(board, move) 