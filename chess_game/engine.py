import random
import chess

class Engine:
    def get_move(self, board):
        raise NotImplementedError

class RandomEngine(Engine):
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
        return None

class MinimaxEngine(Engine):
    def __init__(self, depth=4):
        self.depth = depth

    def get_move(self, board):
        best_move = None
        best_value = -float('inf') if board.turn else float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = self._minimax(board, self.depth - 1, alpha, beta, not board.turn)
            board.pop()
            if board.turn:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
        return best_move

    def _minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return self._quiescence(board, alpha, beta, maximizing)
        if maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, board):
        # Improved material evaluation with better values
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000  # High value to avoid trading king
        }
        # Piece-square tables (center control, etc.)
        pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        knight_table = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]
        bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]
        rook_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            0, 0, 0, 5, 5, 0, 0, 0
        ]
        queen_table = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ]
        king_table = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]
        pst = {
            chess.PAWN: pawn_table,
            chess.KNIGHT: knight_table,
            chess.BISHOP: bishop_table,
            chess.ROOK: rook_table,
            chess.QUEEN: queen_table,
            chess.KING: king_table
        }
        value = 0
        for piece_type in piece_values:
            for square in board.pieces(piece_type, chess.WHITE):
                value += piece_values[piece_type]
                value += pst[piece_type][square]
            for square in board.pieces(piece_type, chess.BLACK):
                value -= piece_values[piece_type]
                value -= pst[piece_type][chess.square_mirror(square)]
        # Add a small bonus for checkmate
        if board.is_checkmate():
            if board.turn:
                value -= 100000
            else:
                value += 100000
        return value

    def _quiescence(self, board, alpha, beta, maximizing):
        stand_pat = self.evaluate(board)
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Only search captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        
        # Sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        captures.sort(key=lambda move: self._get_capture_value(board, move), reverse=True)
        
        for move in captures:
            board.push(move)
            score = self._quiescence(board, alpha, beta, not maximizing)
            board.pop()
            
            if maximizing:
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            else:
                if score < beta:
                    beta = score
                if beta <= alpha:
                    break
        
        return alpha if maximizing else beta
    
    def _get_capture_value(self, board, move):
        """Get the value of a capture move for MVV-LVA sorting"""
        victim_piece = board.piece_at(move.to_square)
        attacker_piece = board.piece_at(move.from_square)
        
        if victim_piece is None or attacker_piece is None:
            return 0
        
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        
        # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
        return piece_values[victim_piece.piece_type] * 10 - piece_values[attacker_piece.piece_type] 