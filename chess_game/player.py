import random

class Player:
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        raise NotImplementedError

class HumanPlayer(Player):
    def get_move(self, board):
        # Human move will be handled by GUI
        return None

class ComputerPlayer(Player):
    def get_move(self, board):
        # Simple random move for now
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
        return None 