import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import chess
import random
import threading
import sys
import select
from chess_game.board import ChessBoard
from chess_game.engine import MinimaxEngine
from chess_game.logging_manager import ChessLoggingManager
import time

ASSET_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'pieces')

class PygameChessGUI:
    def __init__(self, time_budget=None):
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
        
        # Create logging manager for GUI
        logger = ChessLoggingManager(print, quiet=False)
        self.engine = MinimaxEngine(evaluator_type="handcrafted")  # Use config file depth
        # Override the engine's logger to use our GUI logger
        self.engine.logger = logger
        
        # Store time budget for iterative deepening
        self.time_budget = time_budget
        
        # Computer vs Computer mode tracking
        self.computer_vs_computer_mode = False
        self.total_nodes = 0
        self.total_moves = 0
        self.game_start_time = 0
        
        # Terminal input handling
        self.running = True
        self.terminal_input_thread = None
        
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
            else:
                print(f"Image not found: {img_path}")
    
    def setup_checkmate_defense_mode(self):
        """Set up the board for checkmate defense mode: Human (White) has King + Pawn vs Computer (Black) with King + 2 Knights"""
        # Clear the board
        self.board.reset()
        chess_board = self.board.get_board()
        
        # Remove all pieces
        for square in chess.SQUARES:
            if chess_board.piece_at(square):
                chess_board.remove_piece_at(square)
        
        # Place White King (Human) at c4 and White Pawn at g2
        chess_board.set_piece_at(chess.C4, chess.Piece(chess.KING, chess.WHITE))
        chess_board.set_piece_at(chess.G2, chess.Piece(chess.PAWN, chess.WHITE))
        
        # Place Black King at e8 and Black Knights at c6 and g4
        chess_board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        chess_board.set_piece_at(chess.C6, chess.Piece(chess.KNIGHT, chess.BLACK))
        chess_board.set_piece_at(chess.G4, chess.Piece(chess.KNIGHT, chess.BLACK))
        
        print("♔ Checkmate Defense Mode: Human (White King + Pawn) vs Computer (Black King + 2 Knights)")
        print("🎯 Goal: Try to avoid checkmate for as long as possible!")
    
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
        if self.game_mode in ["human_vs_computer", "computer_vs_human", "checkmate_defense", "computer_vs_computer"]:
            chess_board = self.board.get_board()
            if not chess_board.is_game_over():
                # Track nodes for computer vs computer mode
                if self.computer_vs_computer_mode:
                    nodes_before = self.engine.nodes_searched
                    print(f"🤖 Computer vs Computer: {'White' if chess_board.turn else 'Black'} to move")
                
                move = self.engine.get_move(chess_board, time_budget=self.time_budget)
                
                if move:
                    # Update statistics for computer vs computer mode
                    if self.computer_vs_computer_mode:
                        nodes_this_move = self.engine.nodes_searched - nodes_before
                        self.total_nodes += nodes_this_move
                        self.total_moves += 1
                        
                        # Calculate and display average speed
                        elapsed_time = time.time() - self.game_start_time
                        if elapsed_time > 0:
                            avg_nodes_per_second = self.total_nodes / elapsed_time
                            print(f"📊 Game Stats: {self.total_moves} moves, {self.total_nodes} nodes, {avg_nodes_per_second:.0f} avg nodes/s")
                    
                    chess_board.push(move)
                    self.draw_board()
                    
                    # Print FEN after computer move (only for pygame GUI)
                    print(f"🤖 FEN: {chess_board.fen()}")
                    
                    if chess_board.is_game_over():
                        self.show_game_over()
                        
                        # Final statistics for computer vs computer mode
                        if self.computer_vs_computer_mode:
                            self.show_final_stats()
                    else:
                        # In computer vs computer mode, continue with the next move
                        if self.computer_vs_computer_mode:
                            print(f"⏳ Waiting 1 second before next move...")
                            pygame.time.wait(1000)  # Wait 1 second between moves
                            self.make_computer_move()  # Recursive call for next move
    
    def show_final_stats(self):
        """Display final statistics for computer vs computer mode"""
        elapsed_time = time.time() - self.game_start_time
        if elapsed_time > 0:
            avg_nodes_per_second = self.total_nodes / elapsed_time
            print(f"\n🏁 FINAL GAME STATISTICS:")
            print(f"   Total moves: {self.total_moves}")
            print(f"   Total nodes searched: {self.total_nodes:,}")
            print(f"   Total time: {elapsed_time:.2f} seconds")
            print(f"   Average speed: {avg_nodes_per_second:.0f} nodes/second")
            print(f"   Average nodes per move: {self.total_nodes // self.total_moves if self.total_moves > 0 else 0:,}")
    
    def show_game_over(self):
        result = self.board.get_board().result()
        if result == "1-0":
            message = "White wins!"
        elif result == "0-1":
            message = "Black wins!"
        else:
            message = "Draw!"
        
        print(f"Game Over: {message}")
        
        # Special message for checkmate defense mode
        if self.game_mode == "checkmate_defense":
            if result == "0-1":
                print("💀 Checkmate! The computer found a way to checkmate you.")
                print("🎯 Try again to see if you can survive longer!")
            else:
                print("🎉 Amazing! You survived the checkmate attempt!")
        
        # Special message for computer vs computer mode
        if self.computer_vs_computer_mode:
            print("🤖 Computer vs Computer game completed!")
    
    def handle_click(self, pos):
        # In computer vs computer mode, clicks are ignored
        if self.computer_vs_computer_mode:
            return
            
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
            
            # Check if this is a pawn promotion
            piece = self.board.get_board().piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                # Check if pawn is moving to the 8th rank (promotion rank)
                target_rank = chess.square_rank(square)
                if (piece.color == chess.WHITE and target_rank == 7) or (piece.color == chess.BLACK and target_rank == 0):
                    # Create promotion move to queen
                    move = chess.Move(self.selected_square, square, chess.QUEEN)
            
            if move in self.board.get_board().legal_moves:
                self.board.get_board().push(move)
                self.selected_square = None
                self.draw_board()
                
                # Print FEN after human move (only for pygame GUI)
                print(f"📋 FEN: {self.board.get_board().fen()}")
                
                # Check for game over
                if self.board.get_board().is_game_over():
                    self.show_game_over()
                else:
                    # Make computer move if in computer mode
                    if self.game_mode in ["human_vs_computer", "computer_vs_human", "checkmate_defense"]:
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
        
        # Reset computer vs computer statistics
        self.total_nodes = 0
        self.total_moves = 0
        self.game_start_time = time.time()
        
        # Set up special board for checkmate defense mode
        if self.game_mode == "checkmate_defense":
            self.setup_checkmate_defense_mode()
        
        self.draw_board()
        
        # Print initial FEN (only for pygame GUI)
        print(f"🎯 Initial FEN: {self.board.get_board().fen()}")
        
        # If computer plays white, make first move
        if self.game_mode == "computer_vs_human":
            pygame.time.wait(1000)
            self.make_computer_move()
        elif self.game_mode == "computer_vs_computer":
            # Start the computer vs computer game
            pygame.time.wait(1000)
            self.make_computer_move()
    
    def set_game_mode(self, mode):
        self.game_mode = mode
        self.computer_vs_computer_mode = (mode == "computer_vs_computer")
        self.new_game()
        print(f"Game mode set to: {mode}")
        
        # Print special instructions for checkmate defense mode
        if mode == "checkmate_defense":
            print("\n🎯 Checkmate Defense Mode Instructions:")
            print("• You play as White with King + Pawn")
            print("• Computer plays as Black with King + 2 Knights")
            print("• Try to avoid checkmate for as long as possible!")
            print("• Use your King and Pawn to escape and avoid the Knights' attacks")
            print("• The computer will try to checkmate you efficiently")
        
        # Print special instructions for computer vs computer mode
        elif mode == "computer_vs_computer":
            print("\n🤖 Computer vs Computer Mode:")
            print("• Two computer engines will play against each other")
            print("• Search speed and statistics will be tracked")
            print("• Watch the engines analyze positions in real-time")
            print("• Final statistics will be shown at game end")
    
    def handle_terminal_input(self):
        """Handle terminal input in a separate thread"""
        while self.running:
            try:
                # Check if there's input available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        self.process_terminal_command(line)
            except (EOFError, KeyboardInterrupt):
                break
    
    def process_terminal_command(self, command):
        """Process terminal commands"""
        command = command.upper()
        if command == 'N':
            print("🔄 Starting new game...")
            self.new_game()
        elif command == '1':
            print("👥 Switching to Human vs Human mode...")
            self.set_game_mode("human_vs_human")
        elif command == '2':
            print("🤖 Switching to Human vs Computer mode...")
            self.set_game_mode("human_vs_computer")
        elif command == '3':
            print("🤖 Switching to Computer vs Human mode...")
            self.set_game_mode("computer_vs_human")
        elif command == '4':
            print("🎯 Switching to Checkmate Defense mode...")
            self.set_game_mode("checkmate_defense")
        elif command == '5':
            print("🤖 Switching to Computer vs Computer mode...")
            self.set_game_mode("computer_vs_computer")
        elif command == 'EVAL':
            info = self.engine.get_evaluator_info()
            print(f"🧠 Current Evaluator: {info['name']}")
            print(f"📝 Description: {info['description']}")
        elif command == 'SWITCH_EVAL':
            print("Available evaluators:")
            print("  handcrafted - Traditional material + positional evaluation")
            print("  neural - Neural network evaluation (placeholder)")
        elif command == 'ESC' or command == 'QUIT':
            print("👋 Quitting game...")
            self.running = False
        else:
            print(f"❓ Unknown command: {command}")
            print("Available commands: N, 1, 2, 3, 4, 5, EVAL, SWITCH_EVAL, ESC")
    
    def run(self):
        print("Starting Pygame Chess...")
        print("Controls:")
        print("  Mouse: Click to select and move pieces")
        print("  N: New game")
        print("  1: Human vs Human")
        print("  2: Human vs Computer")
        print("  3: Computer vs Human")
        print("  4: Checkmate Defense Mode")
        print("  5: Computer vs Computer")
        print("  E: Show current evaluator info")
        print("  ESC: Quit")
        print("\n💡 Tip: You can type commands in the terminal or use the Pygame window!")
        
        # Start terminal input thread
        self.terminal_input_thread = threading.Thread(target=self.handle_terminal_input, daemon=True)
        self.terminal_input_thread.start()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
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
                    elif event.key == pygame.K_4:
                        self.set_game_mode("checkmate_defense")
                    elif event.key == pygame.K_5:
                        self.set_game_mode("computer_vs_computer")
                    elif event.key == pygame.K_e:
                        info = self.engine.get_evaluator_info()
                        print(f"🧠 Current Evaluator: {info['name']}")
                        print(f"📝 Description: {info['description']}")
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
        
        pygame.quit() 