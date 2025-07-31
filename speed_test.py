#!/usr/bin/env python3

import chess
import time
import sys
import os
import statistics
from typing import Dict, List, Tuple

# Add the chess_game directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'chess_game'))

from engine import MinimaxEngine

class SpeedTester:
    """Comprehensive speed testing for the chess engine"""
    
    def __init__(self):
        self.test_positions = self._get_test_positions()
        self.benchmark_results = {
            'search_speeds': [],
            'evaluation_speeds': [],
            'move_generation_speeds': [],
            'position_performance': [],
            'overall_stats': {}
        }
    
    def _get_test_positions(self) -> List[Tuple[str, str, str]]:
        """Get a variety of test positions for comprehensive benchmarking"""
        return [
            # (name, fen, description)
            ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Standard chess starting position"),
            ("Sicilian Defense", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "After 1.e4 c5"),
            ("Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "After 1.e4 e5 2.Nf3 Nc6 3.Bb5"),
            ("Queen's Gambit", "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4", "After 1.d4 d5 2.c4 e6 3.Nc3 Nf6"),
            ("King's Indian", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "After 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7"),
            ("Endgame KQvK", "8/8/8/8/8/8/4K3/4Q3 w - - 0 1", "Queen vs King endgame"),
            ("Tactical Position", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Tactical position with multiple captures"),
            ("Closed Position", "rnbqkb1r/ppp2ppp/3p1n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 4", "Closed position with limited mobility"),
        ]
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive speed benchmarks"""
        print("🏁 CHESS ENGINE SPEED BENCHMARK")
        print("=" * 50)
        
        # Test different depths
        depths = [2, 3, 4, 5, 6]
        
        # Test different evaluator types
        evaluator_types = ["handcrafted"]  # Add "neural" when implemented
        
        for evaluator_type in evaluator_types:
            print(f"\n🔧 Testing {evaluator_type.upper()} evaluator")
            print("-" * 30)
            
            # Create engine
            engine = MinimaxEngine(depth=4, evaluator_type=evaluator_type)
            
            # Suppress engine output during benchmarks
            import sys
            from io import StringIO
            
            # Redirect stdout to suppress engine output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Test search speed at different depths
                self._benchmark_search_speeds(engine, depths)
                
                # Test evaluation speed
                self._benchmark_evaluation_speed(engine)
                
                # Test move generation speed
                self._benchmark_move_generation(engine)
                
                # Test position-specific performance
                self._benchmark_position_performance(engine, depths[:3])  # Use smaller depths for position tests
            finally:
                # Restore stdout
                sys.stdout = old_stdout
            
            # Calculate and display summary statistics
            self._calculate_summary_statistics()
    
    def _benchmark_search_speeds(self, engine: MinimaxEngine, depths: List[int]):
        """Benchmark search speed at different depths"""
        print(f"🔍 Testing search speeds at depths {depths}...")
        
        # Use starting position for consistent testing
        board = chess.Board()
        
        for depth in depths:
            # Reset engine state
            engine.nodes_searched = 0
            start_time = time.time()
            
            # Perform search
            move = engine.get_move(board)
            
            end_time = time.time()
            search_time = end_time - start_time
            nodes_per_second = engine.nodes_searched / search_time if search_time > 0 else 0
            
            # Store results for summary
            self.benchmark_results['search_speeds'].append({
                'depth': depth,
                'time': search_time,
                'nodes': engine.nodes_searched,
                'nodes_per_second': nodes_per_second,
                'best_move': board.san(move)
            })
    
    def _benchmark_evaluation_speed(self, engine: MinimaxEngine):
        """Benchmark evaluation function speed"""
        print(f"📈 Testing evaluation speed...")
        
        num_evaluations = 10000
        
        for name, fen, description in self.test_positions[:4]:  # Test first 4 positions
            board = chess.Board(fen)
            
            start_time = time.time()
            for _ in range(num_evaluations):
                engine.evaluate(board)
            end_time = time.time()
            
            eval_time = end_time - start_time
            evals_per_second = num_evaluations / eval_time if eval_time > 0 else 0
            
            # Store results for summary
            self.benchmark_results['evaluation_speeds'].append({
                'position': name,
                'evaluations': num_evaluations,
                'time': eval_time,
                'evals_per_second': evals_per_second
            })
    
    def _benchmark_move_generation(self, engine: MinimaxEngine):
        """Benchmark move generation speed"""
        print(f"🎯 Testing move generation speed...")
        
        num_iterations = 1000
        
        for name, fen, description in self.test_positions[:4]:
            board = chess.Board(fen)
            
            start_time = time.time()
            for _ in range(num_iterations):
                legal_moves = list(board.legal_moves)
            end_time = time.time()
            
            move_time_ms = (end_time - start_time) * 1000
            moves_per_ms = len(legal_moves) * num_iterations / move_time_ms if move_time_ms > 0 else 0
            
            # Store results for summary
            self.benchmark_results['move_generation_speeds'].append({
                'position': name,
                'moves': len(legal_moves),
                'time_ms': move_time_ms,
                'moves_per_ms': moves_per_ms
            })
    
    def _benchmark_position_performance(self, engine: MinimaxEngine, depths: List[int]):
        """Benchmark performance across different positions"""
        print(f"🎮 Testing position-specific performance...")
        
        for name, fen, description in self.test_positions:
            board = chess.Board(fen)
            
            for depth in depths:
                # Create new engine instance for each depth to avoid state pollution
                test_engine = MinimaxEngine(depth=depth, evaluator_type="handcrafted")
                test_engine.nodes_searched = 0
                
                start_time = time.time()
                move = test_engine.get_move(board)
                end_time = time.time()
                
                search_time = end_time - start_time
                
                # Store results for summary
                self.benchmark_results['position_performance'].append({
                    'position': name,
                    'depth': depth,
                    'time': search_time,
                    'nodes': test_engine.nodes_searched,
                    'best_move': board.san(move)
                })
    
    def _calculate_summary_statistics(self):
        """Calculate summary statistics"""
        print(f"\n📊 SUMMARY STATISTICS")
        print("=" * 30)
        
        # Search speed
        if self.benchmark_results['search_speeds']:
            search_speeds = [r['nodes_per_second'] for r in self.benchmark_results['search_speeds']]
            print(f"Search speed: {statistics.mean(search_speeds):,.0f} nodes/sec")
        
        # Evaluation speed
        if self.benchmark_results['evaluation_speeds']:
            eval_speeds = [r['evals_per_second'] for r in self.benchmark_results['evaluation_speeds']]
            print(f"Evaluation speed: {statistics.mean(eval_speeds):,.0f} evals/sec")
        
        # Move generation speed
        if self.benchmark_results['move_generation_speeds']:
            move_speeds = [r['moves_per_ms'] for r in self.benchmark_results['move_generation_speeds']]
            print(f"Move generation: {statistics.mean(move_speeds):.1f} moves/ms")
        
        # Overall performance
        if self.benchmark_results['position_performance']:
            all_nodes = [r['nodes'] for r in self.benchmark_results['position_performance']]
            all_times = [r['time'] for r in self.benchmark_results['position_performance']]
            print(f"Total nodes: {sum(all_nodes):,}")
            print(f"Total time: {sum(all_times):.1f}s")
        
        # Store overall stats for potential export
        self.benchmark_results['overall_stats'] = {
            'avg_nodes_per_second': statistics.mean(search_speeds) if self.benchmark_results['search_speeds'] else 0,
            'avg_evals_per_second': statistics.mean(eval_speeds) if self.benchmark_results['evaluation_speeds'] else 0,
            'avg_moves_per_ms': statistics.mean(move_speeds) if self.benchmark_results['move_generation_speeds'] else 0,
            'total_searches': len(self.benchmark_results['position_performance']),
            'total_nodes': sum(all_nodes) if self.benchmark_results['position_performance'] else 0,
            'total_time': sum(all_times) if self.benchmark_results['position_performance'] else 0
        }
    

    
    def run_quick_benchmark(self):
        """Run a quick benchmark for fast testing"""
        print("⚡ QUICK SPEED BENCHMARK")
        print("=" * 30)
        
        engine = MinimaxEngine(depth=4, evaluator_type="handcrafted")
        board = chess.Board()
        
        # Suppress engine output during benchmarks
        import sys
        from io import StringIO
        
        # Redirect stdout to suppress engine output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Quick search test
            engine.nodes_searched = 0
            start_time = time.time()
            move = engine.get_move(board)
            end_time = time.time()
            
            search_time = end_time - start_time
            nodes_per_second = engine.nodes_searched / search_time if search_time > 0 else 0
            
            # Quick evaluation test
            num_evals = 1000
            start_time = time.time()
            for _ in range(num_evals):
                engine.evaluate(board)
            end_time = time.time()
            
            eval_time = end_time - start_time
            evals_per_second = num_evals / eval_time if eval_time > 0 else 0
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Store results for summary
        self.benchmark_results['search_speeds'].append({
            'depth': 4,
            'time': search_time,
            'nodes': engine.nodes_searched,
            'nodes_per_second': nodes_per_second,
            'best_move': board.san(move)
        })
        
        self.benchmark_results['evaluation_speeds'].append({
            'position': 'Quick Test',
            'evaluations': num_evals,
            'time': eval_time,
            'evals_per_second': evals_per_second
        })
        
        # Calculate and display summary
        self._calculate_summary_statistics()

def main():
    """Main speed testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess Engine Speed Testing")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark only")
    args = parser.parse_args()
    
    tester = SpeedTester()
    
    if args.quick:
        tester.run_quick_benchmark()
    else:
        tester.run_comprehensive_benchmark()

if __name__ == "__main__":
    main() 