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
from typing import Optional, List, Tuple

# Add the chess_game directory to the path
sys.path.append('chess_game')
from engine import MinimaxEngine

class TimeoutEngine(MinimaxEngine):
    """Engine with timeout protection and heartbeat pings"""
    
    def __init__(self, depth=4, max_time=30, **kwargs):
        super().__init__(depth, quiet=True, **kwargs)  # Enable quiet mode for UCI
        self.max_time = max_time
        self.search_start_time = 0
        self.last_heartbeat = 0
        self.heartbeat_interval = 2.0  # Send heartbeat every 2 seconds
        # Track best move found so far for timeout fallback
        self.best_move_found = None
        self.best_value_found = None
        self.best_line_found = []
        
    def get_move(self, board):
        """Get move with timeout protection and heartbeat pings"""
        self.search_start_time = time.time()
        self.last_heartbeat = self.search_start_time
        
        # Reset best move tracking
        self.best_move_found = None
        self.best_value_found = None
        self.best_line_found = []
        
        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("Search timed out")
        
        # Set timeout (Unix systems only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.max_time))
        
        try:
            # Print debug message when search starts (proper UCI info command)
            print(f"info string Starting search (depth {self.depth}, max time {self.max_time}s)")
            
            # Use our custom get_move_with_tracking instead of the base class method
            move = self._get_move_with_tracking(board)
            
            # Validate the returned move
            if move is not None and move not in board.legal_moves:
                print(f"info string WARNING: Illegal move {move.uci()} generated, using fallback")
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    return legal_moves[0]
                return None
            
            return move
        except TimeoutError:
            # Return the best move found so far, or first legal move as fallback
            if self.best_move_found is not None:
                print(f"info string Search timed out after {self.max_time}s, using best move found so far (value: {self.best_value_found:.1f})")
                
                # Validate the best move found
                if self.best_move_found in board.legal_moves:
                    return self.best_move_found
                else:
                    print(f"info string WARNING: Best move {self.best_move_found.uci()} is illegal, using fallback")
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        return legal_moves[0]
                    return None
            else:
                print(f"info string Search timed out after {self.max_time}s, using fallback move")
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    return legal_moves[0]
                return None
        finally:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    
    def _minimax(self, board, depth, alpha, beta, variation=None):
        """Minimax with time checking and heartbeat pings"""
        # Check if we're running out of time
        elapsed = time.time() - self.search_start_time
        if elapsed > self.max_time * 0.8:  # Stop at 80% of time limit
            raise TimeoutError("Search taking too long")
        
        # Send heartbeat if enough time has passed
        if elapsed - self.last_heartbeat > self.heartbeat_interval:
            print(f"info string Search in progress... ({elapsed:.1f}s elapsed, depth {depth})")
            self.last_heartbeat = elapsed
        
        return super()._minimax(board, depth, alpha, beta, variation)
    
    def _quiescence(self, board, alpha, beta, depth=0):
        """Quiescence with time checking"""
        # Check if we're running out of time
        elapsed = time.time() - self.search_start_time
        if elapsed > self.max_time * 0.8:
            raise TimeoutError("Quiescence taking too long")
        
        return super()._quiescence(board, alpha, beta, depth)
    
    def _get_move_with_tracking(self, board):
        """
        Custom get_move method that tracks the best move found so far.
        This allows us to return the best move found when a timeout occurs.
        """
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
            print(f"info string Endgame detected - using depth {search_depth}")
        
        best_move = None
        # Initialize best_value based on whose turn it is
        # White wants to maximize (highest value), Black wants to minimize (lowest value)
        best_value = -float('inf') if board_copy.turn else float('inf')
        best_line = []
        alpha = -float('inf')
        beta = float('inf')
        
        # Check if position is in tablebase
        if self.tablebase and self.is_tablebase_position(board_copy):
            print(f"info string Checking tablebase for {sum(len(board_copy.pieces(piece_type, color)) for piece_type in chess.PIECE_TYPES for color in [chess.WHITE, chess.BLACK])} pieces...")
            tablebase_move = self.get_tablebase_move(board_copy)
            if tablebase_move:
                print(f"info string Using tablebase move: {board_copy.san(tablebase_move)}")
                return tablebase_move
            else:
                print(f"info string Tablebase lookup failed, using standard search")
        
        # Start timing the search and reset node counter
        start_time = time.time()
        self.search_start_time = start_time
        self.nodes_searched = 0
        
        # Report search start
        print(f"info string Analyzing {len(list(board_copy.legal_moves))} legal moves...")
        
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
                    print(f"info string New best move: {board_copy.san(move)} (value: {value:.1f})")
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
                    print(f"info string New best move: {board_copy.san(move)} (value: {value:.1f})")
                beta = min(beta, value)
        
        # Report final search results
        if best_move:
            search_time = time.time() - start_time
            nodes_per_second = self.nodes_searched / search_time if search_time > 0 else 0
            print(f"info string Search completed in {search_time:.2f}s | Best: {board_copy.san(best_move)} ({best_value:.1f}) | Speed: {nodes_per_second:.0f} nodes/s")
        
        return best_move

class UCIEngine:
    """UCI-compatible wrapper for the chess engine"""
    
    def __init__(self):
        self.engine = TimeoutEngine(depth=4, max_time=30)
        self.board = chess.Board()
        self.thinking_time = 1000  # milliseconds
        self.depth_limit = 6
        
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
                        self.engine = TimeoutEngine(depth=self.depth_limit, max_time=self.engine.max_time)
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
        
        # Check if this is the first move (starting position with no moves played)
        is_first_move = (self.board.fen().split()[0] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" and 
                        self.board.fullmove_number == 1)
        
        # Calculate thinking time
        if is_first_move:
            # Use 60 seconds for the first move
            thinking_time = 60000
            print(f"info string First move detected - using 60 second timeout")
        elif wtime is not None and btime is not None:
            time_limit = wtime if self.board.turn else btime
            # Use 33% of remaining time with minimum 5 seconds
            thinking_time = max(5000, int(time_limit * 0.33))
        else:
            thinking_time = self.thinking_time
        
        # Update engine with new depth and time limit
        max_time_seconds = thinking_time / 1000.0
        self.engine = TimeoutEngine(depth=depth, max_time=max_time_seconds)
        
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