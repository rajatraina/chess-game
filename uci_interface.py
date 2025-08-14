#!/usr/bin/env python3
"""
UCI (Universal Chess Interface) implementation for the chess engine.
This allows the engine to communicate with chess GUIs and online platforms.
"""

import sys
import chess
import chess.engine
import time
import signal
import os
from datetime import datetime
from typing import Optional, List, Tuple

# Add the chess_game directory to the path
sys.path.append('chess_game')
from engine import MinimaxEngine

class LoggingEngine(MinimaxEngine):
    """Engine wrapper that logs search details to file"""
    
    def __init__(self, depth=None, quiet=True, log_callback=None, **kwargs):
        super().__init__(depth, quiet=True, **kwargs)  # Always quiet for UCI
        self.log_callback = log_callback
        self.search_start_time = 0
        self.best_move_found = None
        self.best_value_found = None
        self.best_line_found = []
        
    def get_move(self, board):
        """Get move with detailed logging"""
        self.search_start_time = time.time()
        self.best_move_found = None
        self.best_value_found = None
        self.best_line_found = []
        
        if self.log_callback:
            self.log_callback(f"Starting search (depth {self.depth})")
            self.log_callback(f"Position: {board.fen()}")
            self.log_callback(f"Legal moves: {len(list(board.legal_moves))}")
        
        # Call the parent method but intercept the search process
        move = self._get_move_with_logging(board)
        
        if self.log_callback and move:
            search_time = time.time() - self.search_start_time
            self.log_callback(f"Search completed in {search_time:.2f}s")
            self.log_callback(f"Best move: {board.san(move)} (value: {self.best_value_found:.1f})")
            self.log_callback(f"Principal variation: {' '.join([board.san(m) for m in self.best_line_found[:5]])}")
            self.log_callback(f"Nodes searched: {self.nodes_searched}")
            self.log_callback(f"Speed: {self.nodes_searched/search_time:.0f} nodes/s")
            self.log_callback("-" * 50)
        
        return move
    
    def _get_move_with_logging(self, board):
        """Custom get_move with detailed logging"""
        # Create a copy of the board to avoid corrupting the original
        board_copy = board.copy()
        
        # Set the starting position for consistent evaluation throughout search
        if hasattr(self.evaluation_manager.evaluator, '_set_starting_position'):
            self.evaluation_manager.evaluator._set_starting_position(board_copy)
        
        # Check if this is an endgame position for deeper search
        is_endgame = self._is_endgame_position(board_copy)
        search_depth = self.depth
        
        if is_endgame:
            endgame_depth = self.evaluation_manager.evaluator.config.get("endgame_search_depth", 6)
            search_depth = max(self.depth, endgame_depth)
            if self.log_callback:
                self.log_callback(f"Endgame detected - using depth {search_depth}")
        
        best_move = None
        # Initialize best_value based on whose turn it is
        # White wants to maximize (highest value), Black wants to minimize (lowest value)
        best_value = -float('inf') if board_copy.turn else float('inf')
        best_line = []
        alpha = -float('inf')
        beta = float('inf')
        
        # Check if position is in tablebase
        if self.tablebase and self.is_tablebase_position(board_copy):
            piece_count = sum(len(board_copy.pieces(piece_type, color)) for piece_type in chess.PIECE_TYPES for color in [chess.WHITE, chess.BLACK])
            if self.log_callback:
                self.log_callback(f"Checking tablebase for {piece_count} pieces...")
            tablebase_move = self.get_tablebase_move(board_copy)
            if tablebase_move:
                if self.log_callback:
                    self.log_callback(f"Using tablebase move: {board_copy.san(tablebase_move)}")
                return tablebase_move
            else:
                if self.log_callback:
                    self.log_callback("Tablebase lookup failed, using standard search")
        
        # Start timing the search and reset node counter
        start_time = time.time()
        self.search_start_time = start_time
        self.nodes_searched = 0
        
        if self.log_callback:
            self.log_callback(f"Analyzing {len(list(board_copy.legal_moves))} legal moves...")
        
        # Evaluate all legal moves
        for move in board_copy.legal_moves:
            # Get the SAN notation before making the move
            move_san = board_copy.san(move)
            
            # Make the move on the board copy
            board_copy.push(move)
            
            try:
                # Search the resulting position
                # Note: After board.push(move), board.turn has changed to the opponent
                value, line = self._minimax(board_copy, search_depth - 1, alpha, beta, [move_san])
            finally:
                # Always undo the move to restore the original board state
                board_copy.pop()
            
            # Update best move based on whose turn it is
            if board_copy.turn:  # White to move: pick highest evaluation
                if value > best_value:
                    best_value = value
                    best_move = move
                    best_line = [move] + line
                    # Update our tracking variables
                    self.best_move_found = best_move
                    self.best_value_found = best_value
                    self.best_line_found = best_line
                    # Report new best move
                    if self.log_callback:
                        self.log_callback(f"New best move: {board_copy.san(move)} (value: {value:.1f})")
                alpha = max(alpha, value)
            else:  # Black to move: pick lowest evaluation
                if value < best_value:
                    best_value = value
                    best_move = move
                    best_line = [move] + line
                    # Update our tracking variables
                    self.best_move_found = best_move
                    self.best_value_found = best_value
                    self.best_line_found = best_line
                    # Report new best move
                    if self.log_callback:
                        self.log_callback(f"New best move: {board_copy.san(move)} (value: {value:.1f})")
                beta = min(beta, value)
        
        return best_move

class UCIEngine:
    """UCI-compatible wrapper for the chess engine"""
    
    def __init__(self):
        self.board = chess.Board()
        self.thinking_time = 1000  # milliseconds
        # Read depth from config file, default to 4 if not found
        try:
            from chess_game.engine import MinimaxEngine
            temp_engine = MinimaxEngine()
            self.depth_limit = temp_engine.depth
        except:
            self.depth_limit = 4  # Fallback default
        self.log_file = "LICHESS-LOG.txt"
        self.move_number = 0
        # Initialize logging engine with callback
        self.engine = LoggingEngine(log_callback=self.log)  # Use config file depth
        
    def log(self, message: str):
        """Log message to file with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
                f.flush()  # Ensure immediate write
        except Exception as e:
            # If logging fails, don't crash the engine
            pass
        
    def run(self):
        """Main UCI loop"""
        print("id name Python Chess Engine")
        print("id author Your Name")
        print("option name Depth type spin default 4 min 1 max 20")
        print("option name ThinkingTime type spin default 1000 min 100 max 30000")
        print("uciok")
        
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                    
                if line == "quit":
                    break
                elif line == "isready":
                    print("readyok")
                elif line == "uci":
                    print("id name Python Chess Engine")
                    print("id author Your Name")
                    print("option name Depth type spin default 4 min 1 max 20")
                    print("option name ThinkingTime type spin default 1000 min 100 max 30000")
                    print("uciok")
                elif line.startswith("setoption"):
                    self._handle_setoption(line)
                elif line.startswith("position"):
                    self._handle_position(line)
                elif line.startswith("go"):
                    self._handle_go(line)
                elif line == "stop":
                    # Engine should stop thinking and return best move
                    pass
                elif line == "quit":
                    break
                    
            except EOFError:
                break
            except Exception as e:
                print(f"info string Error: {e}")
    
    def _handle_setoption(self, line: str):
        """Handle setoption command"""
        parts = line.split()
        if len(parts) >= 4 and parts[1] == "name":
            option_name = parts[2]
            if parts[3] == "value" and len(parts) >= 5:
                value = parts[4]
                
                if option_name == "Depth":
                    try:
                        self.depth_limit = int(value)
                        # Create engine with explicit depth override
                        self.engine = LoggingEngine(depth=self.depth_limit, log_callback=self.log)
                    except ValueError:
                        pass
                elif option_name == "ThinkingTime":
                    try:
                        self.thinking_time = int(value)
                    except ValueError:
                        pass
    
    def _handle_position(self, line: str):
        """Handle position command"""
        parts = line.split()
        if len(parts) < 2:
            return
            
        if parts[1] == "startpos":
            self.board = chess.Board()
            if len(parts) > 2 and parts[2] == "moves":
                for move_str in parts[3:]:
                    try:
                        move = chess.Move.from_uci(move_str)
                        self.board.push(move)
                    except ValueError:
                        pass
        elif parts[1] == "fen":
            if len(parts) >= 3:
                fen = " ".join(parts[2:])
                try:
                    self.board = chess.Board(fen)
                except ValueError:
                    pass
    
    def _handle_go(self, line: str):
        """Handle go command and find best move"""
        parts = line.split()
        
        # Parse time controls
        wtime = btime = winc = binc = movestogo = None
        depth = self.depth_limit
        
        i = 1
        while i < len(parts):
            if parts[i] == "wtime" and i + 1 < len(parts):
                wtime = int(parts[i + 1])
            elif parts[i] == "btime" and i + 1 < len(parts):
                btime = int(parts[i + 1])
            elif parts[i] == "winc" and i + 1 < len(parts):
                winc = int(parts[i + 1])
            elif parts[i] == "binc" and i + 1 < len(parts):
                binc = int(parts[i + 1])
            elif parts[i] == "movestogo" and i + 1 < len(parts):
                movestogo = int(parts[i + 1])
            elif parts[i] == "depth" and i + 1 < len(parts):
                depth = int(parts[i + 1])
            i += 2
        
        # Update engine with new depth (use config file depth if not explicitly set)
        if depth == self.depth_limit:
            # Use config file depth
            self.engine = LoggingEngine(log_callback=self.log)
        else:
            # Use explicit depth override
            self.engine = LoggingEngine(depth=depth, log_callback=self.log)
        
        # Increment move number and log move start
        self.move_number += 1
        self.log(f"=== MOVE {self.move_number} ===")
        self.log(f"Side to move: {'White' if self.board.turn else 'Black'}")
        self.log(f"Search depth: {depth}")
        
        # Find best move
        start_time = time.time()
        best_move = self.engine.get_move(self.board)
        elapsed = (time.time() - start_time) * 1000
        
        # Final validation before sending move
        if best_move and best_move in self.board.legal_moves:
            print(f"info string Playing move: {self.board.san(best_move)}")
            print(f"bestmove {best_move.uci()}")
        else:
            if best_move:
                print(f"info string ERROR: Illegal move {best_move.uci()} generated, resigning")
            print("bestmove 0000")  # Resign

def main():
    """Main entry point"""
    engine = UCIEngine()
    engine.run()

if __name__ == "__main__":
    main() 