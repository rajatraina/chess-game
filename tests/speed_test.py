#!/usr/bin/env python3
"""
Speed testing for the chess engine using a single FEN position.
"""

import chess
import time
from chess_game.engine import MinimaxEngine

class SpeedTester:
    """Speed testing for the chess engine using iterative deepening on a single position"""
    
    def __init__(self):
        self.benchmark_results = {
            'search_speeds': [],
            'evaluation_speeds': [],
            'move_generation_speeds': [],
            'position_performance': []
        }
    
    def run_quick_benchmark(self):
        """Run speed benchmark using iterative deepening on a single position"""
        print("⚡ SPEED BENCHMARK - ITERATIVE DEEPENING")
        print("=" * 50)
        
        # Create engine with opening book disabled and logging enabled
        engine = MinimaxEngine(evaluator_type="handcrafted", quiet=False)
        engine.opening_book = None  # Disable opening book
        
        # Use the specified FEN position
        test_fen = "r2qkb1r/5pp1/p1n1pn2/1p1p2B1/B2P2b1/2N2N2/PP2QPPP/R4RK1 b kq - 1 10"
        board = chess.Board(test_fen)
        
        print(f"🎯 Test Position: {test_fen}")
        print(f"📋 Board:")
        print(board)
        print()
        
        try:
            # Reset counters
            engine.nodes_searched = 0
            
            # Run iterative deepening search using get_move (which calls _iterative_deepening_search internally)
            print("🔍 Running iterative deepening search...")
            start_time = time.perf_counter()
            move = engine.get_move(board)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            nodes_per_second = int(engine.nodes_searched / elapsed) if elapsed > 0 else 0
            
            print(f"✅ Search completed:")
            print(f"  Time: {elapsed:.3f} seconds")
            print(f"  Nodes: {engine.nodes_searched:,}")
            print(f"  Speed: {nodes_per_second:,} nodes/second")
            print(f"  Best move: {board.san(move) if move else 'None'}")
            
            # Get killer move statistics
            killer_stats = engine.get_killer_move_stats()
            print(f"  Killer moves stored: {killer_stats['killer_moves_stored']}")
            print(f"  Killer moves used: {killer_stats['killer_moves_used']}")
            print(f"  Total cutoffs: {killer_stats['total_cutoffs']}")
            
            # Print killer moves data by depth
            print("\n📊 Killer Moves Data:")
            if engine.killer_moves:
                for depth in sorted(engine.killer_moves.keys()):
                    killer_dict = engine.killer_moves[depth]
                    if killer_dict:
                        print(f"  depth={depth} ", end="")
                        # Sort by cutoff count (highest first) for display
                        sorted_killers = sorted(killer_dict.items(), key=lambda x: x[1], reverse=True)
                        killer_moves_str = " ".join([f"{board.san(move)} [{count}]" for move, count in sorted_killers])
                        print(killer_moves_str)
            else:
                print("  No killer moves stored")
            
            # Store results
            self.benchmark_results['search_speeds'].append({
                'time': elapsed,
                'nodes': engine.nodes_searched,
                'nodes_per_second': nodes_per_second,
                'move': str(move),
                'killer_stats': killer_stats
            })
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            raise

if __name__ == "__main__":
    tester = SpeedTester()
    tester.run_quick_benchmark()