import chess

class ChessBoard:
    def __init__(self):
        self.board = chess.Board()

    def move(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def get_board(self):
        return self.board

    def reset(self):
        self.board.reset() 