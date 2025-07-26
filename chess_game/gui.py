import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import chess
import random
from chess_game.board import ChessBoard
from chess_game.player import HumanPlayer, ComputerPlayer

ASSET_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'pieces')

class ChessGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Python Chess')
        
        # Force window to front and make it visible
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        self.board = ChessBoard()
        self.canvas = tk.Canvas(self.root, width=640, height=640)
        self.canvas.pack()
        
        # Add a simple label to verify window content
        test_label = tk.Label(self.root, text="Chess Game - If you see this, the window is working!", 
                             font=("Arial", 16), fg="blue")
        test_label.pack()
        
        # Test if canvas is working
        self.canvas.create_rectangle(0, 0, 100, 100, fill='red', outline='black')
        print("Test rectangle created")  # DEBUG
        
        self.images = {}
        self.selected_square = None
        self.game_mode = "human_vs_human"  # Default mode
        self.load_images()
        
        try:
            self.draw_board()
        except Exception as e:
            print(f"Error in draw_board: {e}")  # DEBUG
        
        self.setup_click_handlers()
        self.create_menu()

    def load_images(self):
        pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk',
                  'bp', 'bn', 'bb', 'br', 'bq', 'bk']
        for piece in pieces:
            img_path = os.path.join(ASSET_PATH, f'{piece}.png')
            print(f"Loading image for {piece}: {img_path}")  # DEBUG
            if os.path.exists(img_path):
                self.images[piece] = ImageTk.PhotoImage(Image.open(img_path).resize((80, 80)))
                print(f"Loaded {piece}")  # DEBUG
            else:
                print(f"Image not found: {img_path}")  # DEBUG

    def draw_board(self):
        print("Drawing board...")  # DEBUG
        colors = ['#f0d9b5', '#b58863']
        for row in range(8):
            for col in range(8):
                x1 = col * 80
                y1 = row * 80
                x2 = x1 + 80
                y2 = y1 + 80
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        print("Board squares drawn, now drawing pieces...")  # DEBUG
        self.draw_pieces()

    def draw_pieces(self):
        print("Starting to draw pieces...")  # DEBUG
        # Clear existing pieces
        for item in self.canvas.find_all():
            if self.canvas.type(item) == 'image':
                self.canvas.delete(item)
        
        # Store references to PhotoImage objects
        self.piece_images_on_board = []
        
        # Draw pieces based on current board state
        chess_board = self.board.get_board()
        print(f"Chess board state: {chess_board}")  # DEBUG
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
                print(f"Drawing piece {image_key} at square {square} (col={col}, row={row})")  # DEBUG
                if image_key in self.images:
                    x = col * 80 + 40
                    y = row * 80 + 40
                    img_obj = self.images[image_key]
                    self.canvas.create_image(x, y, image=img_obj)
                    self.piece_images_on_board.append(img_obj)  # Keep reference
                    print(f"Created image for {image_key} at ({x}, {y})")  # DEBUG
                else:
                    print(f"Image key {image_key} not found in self.images")  # DEBUG
        print("Finished drawing pieces")  # DEBUG

    def setup_click_handlers(self):
        self.canvas.bind('<Button-1>', self.on_square_click)

    def on_square_click(self, event):
        col = event.x // 80
        row = event.y // 80
        square = chess.square(col, 7 - row)  # Convert to chess square
        
        if self.selected_square is None:
            # Select piece
            piece = self.board.get_board().piece_at(square)
            if piece and piece.color == self.board.get_board().turn:
                self.selected_square = square
                self.highlight_square(col, row)
        else:
            # Try to make move
            move = chess.Move(self.selected_square, square)
            if move in self.board.get_board().legal_moves:
                self.board.get_board().push(move)
                self.draw_board()
                self.selected_square = None
                self.clear_highlights()
                
                # Check for game over
                if self.board.get_board().is_game_over():
                    self.show_game_over()
                else:
                    # Make computer move if in computer mode
                    self.root.after(500, self.make_computer_move)
            else:
                # Invalid move, try to select different piece
                piece = self.board.get_board().piece_at(square)
                if piece and piece.color == self.board.get_board().turn:
                    self.selected_square = square
                    self.clear_highlights()
                    self.highlight_square(col, row)
                else:
                    self.selected_square = None
                    self.clear_highlights()

    def highlight_square(self, col, row):
        x1 = col * 80
        y1 = row * 80
        x2 = x1 + 80
        y2 = y1 + 80
        self.canvas.create_rectangle(x1, y1, x2, y2, outline='yellow', width=3)

    def clear_highlights(self):
        for item in self.canvas.find_all():
            if self.canvas.type(item) == 'rectangle' and len(self.canvas.find_withtag(item)) > 0:
                coords = self.canvas.coords(item)
                if len(coords) == 4 and coords[2] - coords[0] == 80 and coords[3] - coords[1] == 80:
                    # This is a square, check if it has yellow outline
                    outline = self.canvas.itemcget(item, 'outline')
                    if outline == 'yellow':
                        self.canvas.delete(item)

    def show_game_over(self):
        result = self.board.get_board().result()
        if result == "1-0":
            message = "White wins!"
        elif result == "0-1":
            message = "Black wins!"
        else:
            message = "Draw!"
        
        tk.messagebox.showinfo("Game Over", message)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.new_game)
        game_menu.add_separator()
        game_menu.add_command(label="Human vs Human", command=lambda: self.set_game_mode("human_vs_human"))
        game_menu.add_command(label="Human vs Computer", command=lambda: self.set_game_mode("human_vs_computer"))
        game_menu.add_command(label="Computer vs Human", command=lambda: self.set_game_mode("computer_vs_human"))
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)

    def set_game_mode(self, mode):
        self.game_mode = mode
        self.new_game()
        messagebox.showinfo("Game Mode", f"Mode set to: {mode.replace('_', ' ').title()}")

    def make_computer_move(self):
        if self.game_mode in ["human_vs_computer", "computer_vs_human"]:
            chess_board = self.board.get_board()
            if not chess_board.is_game_over():
                # Simple random move
                legal_moves = list(chess_board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    chess_board.push(move)
                    self.draw_board()
                    
                    if chess_board.is_game_over():
                        self.show_game_over()

    def new_game(self):
        self.board.reset()
        self.selected_square = None
        self.clear_highlights()
        self.draw_board()
        
        # If computer plays white, make first move
        if self.game_mode == "computer_vs_human":
            self.root.after(1000, self.make_computer_move)

    def run(self):
        self.root.mainloop() 