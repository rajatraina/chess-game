#!/usr/bin/env python3
"""
Simple profiling script for the chess engine search function.
"""

import cProfile
import pstats
import sys
import os
import time

# Add the chess_game directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chess_game'))

import chess
from engine import MinimaxEngine

def profile_search():
    """Profile the main search function"""
    
    # Test position from user
    fen = "r2qkb1r/ppp2ppp/2n5/3npb2/P6P/2PP2P1/1B1NPPB1/R2QK1NR b KQkq - 2 9"
    time_budget = 30.0  # 30 seconds
    
    print(f"🎯 PROFILING SEARCH FUNCTION")
    print(f"{'='*60}")
    print(f"FEN: {fen}")
    print(f"Time budget: {time_budget} seconds")
    
    # Create board and engine
    board = chess.Board(fen)
    engine = MinimaxEngine(evaluator_type="handcrafted")
    
    # Set starting position
    if hasattr(engine.evaluation_manager.evaluator, '_set_starting_position'):
        engine.evaluation_manager.evaluator._set_starting_position(board)
    
    # Profile the search
    print(f"\n🚀 Starting search with profiling...")
    start_time = time.time()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # This is the main function we want to profile
    best_move = engine.get_move(board, time_budget=time_budget)
    
    profiler.disable()
    end_time = time.time()
    
    search_time = end_time - start_time
    
    print(f"\n✅ Search completed!")
    print(f"Search time: {search_time:.2f} seconds")
    print(f"Best move: {board.san(best_move) if best_move else 'None'}")
    print(f"Nodes searched: {engine.nodes_searched:,}")
    print(f"Nodes per second: {engine.nodes_searched / search_time:,.0f}")
    
    # Save profile
    profile_file = "search_profile.prof"
    profiler.dump_stats(profile_file)
    print(f"\n📊 Profile saved to: {profile_file}")
    
    # Analyze profile
    analyze_profile(profile_file)

def analyze_profile(profile_file):
    """Analyze and display profile results"""
    stats = pstats.Stats(profile_file)
    
    print(f"\n{'='*60}")
    print(f"PROFILE ANALYSIS")
    print(f"{'='*60}")
    
    # Get total stats
    total_time = stats.total_tt
    total_calls = stats.total_calls
    
    print(f"Total function calls: {total_calls:,}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per call: {total_time / total_calls * 1000:.3f} ms")
    
    # Top functions by cumulative time
    print(f"\n📊 TOP 15 FUNCTIONS BY CUMULATIVE TIME:")
    print(f"{'='*60}")
    stats.sort_stats('cumulative')
    stats.print_stats(15)
    
    # Top functions by self time
    print(f"\n📊 TOP 15 FUNCTIONS BY SELF TIME:")
    print(f"{'='*60}")
    stats.sort_stats('tottime')
    stats.print_stats(15)
    
    # Search-specific functions
    print(f"\n🔍 SEARCH FUNCTIONS:")
    print(f"{'='*60}")
    stats.sort_stats('cumulative')
    stats.print_stats('get_move|_minimax|_quiescence|_iterative_deepening')
    
    # Evaluation functions
    print(f"\n🔍 EVALUATION FUNCTIONS:")
    print(f"{'='*60}")
    stats.sort_stats('cumulative')
    stats.print_stats('evaluate|_evaluate_')
    
    # Chess library functions
    print(f"\n🔍 CHESS LIBRARY FUNCTIONS:")
    print(f"{'='*60}")
    stats.sort_stats('cumulative')
    stats.print_stats('chess/__init__.py')

if __name__ == "__main__":
    try:
        profile_search()
    except KeyboardInterrupt:
        print(f"\n⚠️  Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
