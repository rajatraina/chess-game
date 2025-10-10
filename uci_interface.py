#!/usr/bin/env python3
"""
UCI-compatible chess engine interface with improved logging.
"""

import chess
import time
from datetime import datetime
from chess_game.engine import MinimaxEngine
from chess_game.logging_manager import ChessLoggingManager

class LoggingEngine(MinimaxEngine):
    """Engine with logging capabilities using unified logging manager"""
    
    def __init__(self, depth=None, log_callback=None):
        # Create a logging manager for this engine
        logger = ChessLoggingManager(log_callback, quiet=False)
        super().__init__(depth=depth, new_best_move_callback=log_callback)
        # Override the logger to use our custom one
        self.logger = logger

class UCIEngine:
    """UCI-compatible wrapper for the chess engine"""
    
    def __init__(self):
        self.board = chess.Board()
        self.thinking_time = 1000  # milliseconds
        self.move_history = []  # Track move history for repetition detection
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
        print("id name ChessEngine")
        print("id author Your Name")
        
        # Declare supported UCI options
        print("option name Depth type spin default 2 min 1 max 10")
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
                elif line.startswith("position"):
                    self._handle_position(line)
                elif line.startswith("go"):
                    self._handle_go(line)
                elif line.startswith("setoption"):
                    self._handle_setoption(line)
                elif line == "ucinewgame":
                    self.board = chess.Board()
                    self.move_number = 0
                    self.move_history = []  # Reset move history
                    
            except EOFError:
                break
            except Exception as e:
                print(f"info string Error: {e}")
    
    def _handle_setoption(self, line: str):
        """Handle setoption command"""
        parts = line.split()
        if len(parts) >= 4 and parts[1] == "name" and parts[3] == "value":
            option_name = parts[2]
            value = parts[4] if len(parts) > 4 else ""
            
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
            self.move_history = []  # Reset move history
            if len(parts) > 2 and parts[2] == "moves":
                for move_str in parts[3:]:
                    try:
                        move = chess.Move.from_uci(move_str)
                        self.board.push(move)
                        self.move_history.append(move)  # Track move history
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
        
        # Only recreate engine if depth actually changed
        if not hasattr(self, '_current_depth') or self._current_depth != depth:
            if depth == self.depth_limit:
                # Use config file depth
                self.engine = LoggingEngine(log_callback=self.log)
            else:
                # Use explicit depth override
                self.engine = LoggingEngine(depth=depth, log_callback=self.log)
            self._current_depth = depth
        
        # Increment move number and log move start
        self.move_number += 1
        self.log(f"=== MOVE {self.move_number} ===")
        
        # Log time remaining stats
        time_stats = []
        if wtime is not None:
            time_stats.append(f"White: {wtime/1000:.1f}s")
        if btime is not None:
            time_stats.append(f"Black: {btime/1000:.1f}s")
        if winc is not None:
            time_stats.append(f"White inc: {winc/1000:.1f}s")
        if binc is not None:
            time_stats.append(f"Black inc: {binc/1000:.1f}s")
        if movestogo is not None:
            time_stats.append(f"Moves to go: {movestogo}")
        
        if time_stats:
            self.log(f"Time remaining: {', '.join(time_stats)}")
        else:
            self.log("Time remaining: No time info provided")
        
        self.log(f"Side to move: {'White' if self.board.turn else 'Black'}")
        self.log(f"Search depth: {depth}")
        self.log(f"Board FEN: {self.board.fen()}")
        
        # Show recent move history (last 10 moves)
        if len(self.board.move_stack) > 0:
            recent_moves = []
            for i, move in enumerate(self.board.move_stack[-10:]):
                # Create a temporary board to get SAN notation
                temp_board = chess.Board()
                for j, prev_move in enumerate(self.board.move_stack[:len(self.board.move_stack)-10+i]):
                    temp_board.push(prev_move)
                try:
                    san_move = temp_board.san(move)
                    recent_moves.append(san_move)
                except:
                    recent_moves.append(move.uci())
            self.log(f"Recent moves: {' '.join(recent_moves)}")
        
        # Check for draw conditions
        repetition_detected = False
        if self.board.is_repetition():
            self.log("⚠️ 3-fold repetition detected!")
            repetition_detected = True
        elif self._check_repetition_with_history():
            self.log("⚠️ 3-fold repetition detected (using move history)!")
            repetition_detected = True
        if self.board.is_fifty_moves():
            self.log("⚠️ 50-move rule draw detected!")
        if self.board.is_insufficient_material():
            self.log("⚠️ Insufficient material draw detected!")
        if self.board.is_stalemate():
            self.log("⚠️ Stalemate detected!")
        if self.board.is_checkmate():
            self.log("⚠️ Checkmate detected!")
        
        # Calculate time budget
        time_budget = None
        if self.board.turn and wtime is not None:  # White's turn
            time_budget = self.engine.calculate_time_budget(wtime, winc or 0, self.board)
        elif not self.board.turn and btime is not None:  # Black's turn
            time_budget = self.engine.calculate_time_budget(btime, binc or 0, self.board)
        
        if time_budget is not None:
            self.engine.logger.log_time_budget(time_budget)
        
        # Find best move
        start_time = time.time()
        best_move = self.engine.get_move(self.board, time_budget, repetition_detected)
        elapsed = (time.time() - start_time) * 1000
        
        # Determine if search completed or was interrupted
        search_completed = not getattr(self.engine, 'search_interrupted', False)
        
        # Final validation before sending move
        if best_move and best_move in self.board.legal_moves:
            try:
                move_san = self.board.san(best_move)
                self.engine.logger.log_move_sent(move_san, search_completed)
                print(f"info string Playing move: {move_san}")
            except Exception as e:
                self.engine.logger.log_move_sent(best_move.uci(), search_completed)
                print(f"info string Playing move: {best_move.uci()} (SAN error: {e})")
            print(f"bestmove {best_move.uci()}")
        else:
            if best_move:
                self.engine.logger.log_error(f"Illegal move {best_move.uci()} generated, resigning")
                print(f"info string ERROR: Illegal move {best_move.uci()} generated, resigning")
            else:
                self.engine.logger.log_error("No move generated, resigning")
            print("bestmove 0000")  # Resign

    def _check_repetition_with_history(self):
        """Check for 3-fold repetition using move history"""
        if len(self.move_history) < 6:  # Need at least 6 moves for 3-fold repetition
            return False
        
        # Create a board and replay moves to check for repetition
        temp_board = chess.Board()
        position_count = {}
        
        # Count the starting position
        position_count[temp_board._transposition_key()] = 1
        
        for move in self.move_history:
            temp_board.push(move)
            pos_key = temp_board._transposition_key()
            position_count[pos_key] = position_count.get(pos_key, 0) + 1
            
            # Check if current position has occurred 3 times
            if position_count[pos_key] >= 3:
                return True
        
        return False

def main():
    """Main entry point"""
    engine = UCIEngine()
    engine.run()

if __name__ == "__main__":
    main()
