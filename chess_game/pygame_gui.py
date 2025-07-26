import pygame
import os
import chess
import random
from chess_game.board import ChessBoard

ASSET_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'pieces')

class PygameChessGUI:
    def __init__(self):
        pygame.init()
        self.square_size = 60  # Use native image size
        self.width = self.square_size * 8
        self.height = self.square_size * 8
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Python Chess - Pygame')
        
        self.board = ChessBoard()
        self.images = {}
        self.selected_square = None
        self.game_mode = "human_vs_computer"
        
        self.load_images()
        self.draw_board()
        
    def load_images(self):
        pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk',
                  'bp', 'bn', 'bb', 'br', 'bq', 'bk']
        for piece in pieces:
            img_path = os.path.join(ASSET_PATH, f'{piece}.png')
            if os.path.exists(img_path):
                img = pygame.image.load(img_path)
                self.images[piece] = img  # Use native size
                print(f"Loaded {piece}")
            else:
                print(f"Image not found: {img_path}")
    
    def draw_board(self):
        # Draw board squares
        colors = [(240, 217, 181), (181, 136, 99)]  # Light and dark squares
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
        
        # Draw pieces
        chess_board = self.board.get_board()
        for square in chess.SQUARES:
            piece = chess_board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                col = file
                row = 7 - rank  # Flip for display
                
                piece_symbol = piece.symbol()
                color = 'w' if piece.color else 'b'
                piece_type = piece_symbol.upper()
                
                # Map piece types to image keys
                piece_map = {
                    'P': 'p', 'N': 'n', 'B': 'b', 'R': 'r', 'Q': 'q', 'K': 'k'
                }
                
                image_key = color + piece_map[piece_type]
                if image_key in self.images:
                    x = col * self.square_size
                    y = row * self.square_size
                    self.screen.blit(self.images[image_key], (x, y))
        
        # Highlight selected square
        if self.selected_square is not None:
            file = chess.square_file(self.selected_square)
            rank = chess.square_rank(self.selected_square)
            col = file
            row = 7 - rank
            x = col * self.square_size
            y = row * self.square_size
            pygame.draw.rect(self.screen, (255, 255, 0), (x, y, self.square_size, self.square_size), 3)
        
        pygame.display.flip()
    
    def get_square_from_pos(self, pos):
        x, y = pos
        col = x // self.square_size
        row = y // self.square_size
        if 0 <= col <= 7 and 0 <= row <= 7:
            return chess.square(col, 7 - row)
        return None
    
    def make_computer_move(self):
        if self.game_mode in ["human_vs_computer", "computer_vs_human"]:
            chess_board = self.board.get_board()
            if not chess_board.is_game_over():
                legal_moves = list(chess_board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    chess_board.push(move)
                    self.draw_board()
                    
                    if chess_board.is_game_over():
                        self.show_game_over()
    
    def show_game_over(self):
        result = self.board.get_board().result()
        if result == "1-0":
            message = "White wins!"
        elif result == "0-1":
            message = "Black wins!"
        else:
            message = "Draw!"
        
        print(f"Game Over: {message}")
    
    def handle_click(self, pos):
        square = self.get_square_from_pos(pos)
        if square is None:
            return
        
        if self.selected_square is None:
            # Select piece
            piece = self.board.get_board().piece_at(square)
            if piece and piece.color == self.board.get_board().turn:
                self.selected_square = square
                self.draw_board()
        else:
            # Try to make move
            move = chess.Move(self.selected_square, square)
            if move in self.board.get_board().legal_moves:
                self.board.get_board().push(move)
                self.selected_square = None
                self.draw_board()
                
                # Check for game over
                if self.board.get_board().is_game_over():
                    self.show_game_over()
                else:
                    # Make computer move if in computer mode
                    if self.game_mode in ["human_vs_computer", "computer_vs_human"]:
                        pygame.time.wait(500)
                        self.make_computer_move()
            else:
                # Invalid move, try to select different piece
                piece = self.board.get_board().piece_at(square)
                if piece and piece.color == self.board.get_board().turn:
                    self.selected_square = square
                    self.draw_board()
                else:
                    self.selected_square = None
                    self.draw_board()
    
    def new_game(self):
        self.board.reset()
        self.selected_square = None
        self.draw_board()
        
        # If computer plays white, make first move
        if self.game_mode == "computer_vs_human":
            pygame.time.wait(1000)
            self.make_computer_move()
    
    def set_game_mode(self, mode):
        self.game_mode = mode
        self.new_game()
        print(f"Game mode set to: {mode}")
    
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        self.new_game()
                    elif event.key == pygame.K_1:
                        self.set_game_mode("human_vs_human")
                    elif event.key == pygame.K_2:
                        self.set_game_mode("human_vs_computer")
                    elif event.key == pygame.K_3:
                        self.set_game_mode("computer_vs_human")
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            clock.tick(60)
        
        pygame.quit() 