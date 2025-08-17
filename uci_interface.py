#!/usr/bin/env python3
"""
UCI-compatible chess engine interface with improved logging.
"""

import chess
import time
from datetime import datetime
from chess_game.engine import MinimaxEngine

class LoggingEngine(MinimaxEngine):
    """Engine with logging capabilities"""
    
    def __init__(self, depth=None, log_callback=None):
        super().__init__(depth=depth, new_best_move_callback=log_callback)
        self.log_callback = log_callback or print
        # Suppress stdout logging when running as UCI engine
        self.quiet = True
    
    def get_move(self, board, time_budget=None):
        """Get best move with logging"""
        if self.log_callback:
            self.log_callback("🤔 Engine thinking...")
        
        # Reset search stats
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        # Get best move
        best_move = super().get_move(board, time_budget)
        
        # Log search results
        if self.log_callback:
            elapsed = time.time() - self.search_start_time
            speed = self.nodes_searched / elapsed if elapsed > 0 else 0
            
            # Get TT stats
            tt_hits = getattr(self, 'tt_hits', 0)
            tt_misses = getattr(self, 'tt_misses', 0)
            tt_cutoffs = getattr(self, 'tt_cutoffs', 0)
            total_tt = tt_hits + tt_misses
            tt_hit_rate = (tt_hits / total_tt * 100) if total_tt > 0 else 0
            
            # Log search completion
            self.log_callback(f"⏱️ Search completed in {elapsed:.2f}s")
            self.log_callback(f"🔄 TT: {tt_hits}/{total_tt} hits ({tt_hit_rate:.1f}%) | Cutoffs: {tt_cutoffs} ({tt_cutoffs/tt_hits*100:.1f}%)" if tt_hits > 0 else "🔄 TT: No hits")
            
            # Log best move and principal variation
            if best_move:
                try:
                    move_san = board.san(best_move)
                    self.log_callback(f"🏆 Best: {move_san}")
                    
                    # Log principal variation if available
                    if hasattr(self, 'best_line_found') and self.best_line_found:
                        try:
                            pv_moves = []
                            pv_board = board.copy()
                            for m in self.best_line_found[:5]:  # Show first 5 moves
                                if m in pv_board.legal_moves:
                                    pv_moves.append(pv_board.san(m))
                                    pv_board.push(m)
                                else:
                                    break
                            pv_string = ' '.join(pv_moves) if pv_moves else "N/A"
                            self.log_callback(f"📊 PV: {pv_string}")
                        except Exception as e:
                            self.log_callback(f"📊 PV: Error generating PV: {e}")
                    
                    self.log_callback(f"🚀 Speed: {speed:.0f} nodes/s")
                    
                    # Log evaluation components if available
                    try:
                        eval_components = self.evaluate_with_components(board)
                        if eval_components:
                            material = eval_components.get('material', 0)
                            positional = eval_components.get('positional', 0)
                            mobility = eval_components.get('mobility', 0)
                            overall_eval = material + positional + mobility
                            self.log_callback(f"📊 Overall: {overall_eval:.1f} (Material: {material:.1f}, Position: {positional:.1f}, Mobility: {mobility:.1f})")
                    except Exception:
                        pass
                        
                except Exception as e:
                    self.log_callback(f"🏆 Best: {best_move.uci()} (SAN error: {e})")
        
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
        
        # Calculate time budget
        time_budget = None
        if self.board.turn and wtime is not None:  # White's turn
            time_budget = self.engine.calculate_time_budget(wtime, winc or 0)
        elif not self.board.turn and btime is not None:  # Black's turn
            time_budget = self.engine.calculate_time_budget(btime, binc or 0)
        
        if time_budget is not None:
            self.log(f"Time budget: {time_budget:.2f}s")
        
        # Find best move
        start_time = time.time()
        best_move = self.engine.get_move(self.board, time_budget)
        elapsed = (time.time() - start_time) * 1000
        
        # Determine if search completed or was interrupted
        search_completed = not getattr(self.engine, 'search_interrupted', False)
        
        # Final validation before sending move
        if best_move and best_move in self.board.legal_moves:
            try:
                move_san = self.board.san(best_move)
                move_status = "completed" if search_completed else "timeout"
                self.log(f"🎯 Move sent: {move_san} ({move_status})")
                print(f"info string Playing move: {move_san}")
            except Exception as e:
                move_status = "completed" if search_completed else "timeout"
                self.log(f"🎯 Move sent: {best_move.uci()} ({move_status}, SAN error: {e})")
                print(f"info string Playing move: {best_move.uci()} (SAN error: {e})")
            print(f"bestmove {best_move.uci()}")
        else:
            if best_move:
                self.log(f"❌ ERROR: Illegal move {best_move.uci()} generated, resigning")
                print(f"info string ERROR: Illegal move {best_move.uci()} generated, resigning")
            else:
                self.log(f"❌ ERROR: No move generated, resigning")
            print("bestmove 0000")  # Resign

def main():
    """Main entry point"""
    engine = UCIEngine()
    engine.run()

if __name__ == "__main__":
    main()
