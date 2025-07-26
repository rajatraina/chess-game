class Engine:
    def get_move(self, board):
        raise NotImplementedError

class RandomEngine(Engine):
    def get_move(self, board):
        import random
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
        return None 