#!/usr/bin/env python3
"""
UCI (Universal Chess Interface) implementation for the chess engine.
This allows the engine to communicate with chess GUIs and online platforms.
"""

import sys
import chess
import chess.engine
import time
from typing import Optional, List, Tuple

# Add the chess_game directory to the path
sys.path.append('chess_game')
from engine import MinimaxEngine

class UCIEngine:
    """UCI-compatible wrapper for the chess engine"""
    
    def __init__(self):
        self.engine = MinimaxEngine(depth=4)
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
                        self.engine = MinimaxEngine(depth=self.depth_limit)
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
        
        # Calculate thinking time
        if wtime is not None and btime is not None:
            time_limit = wtime if self.board.turn else btime
            if movestogo:
                thinking_time = max(100, time_limit // (movestogo * 10))
            else:
                thinking_time = max(100, time_limit // 30)
        else:
            thinking_time = self.thinking_time
        
        # Find best move
        start_time = time.time()
        best_move = self.engine.get_move(self.board)
        elapsed = (time.time() - start_time) * 1000
        
        if best_move:
            print(f"bestmove {best_move.uci()}")
        else:
            print("bestmove 0000")  # Resign

def main():
    """Main entry point"""
    engine = UCIEngine()
    engine.run()

if __name__ == "__main__":
    main() 