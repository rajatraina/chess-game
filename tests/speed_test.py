#!/usr/bin/env python3
"""
Speed testing for the chess engine.
"""

import chess
import time
import statistics
from typing import List, Tuple
from chess_game.engine import MinimaxEngine

class SpeedTester:
    """Comprehensive speed testing for the chess engine"""
    
    def __init__(self):
        self.test_positions = self._get_test_positions()
        self.benchmark_results = {
            'search_speeds': [],
            'evaluation_speeds': [],
            'move_generation_speeds': [],
            'position_performance': []
        }
    
    def _get_test_positions(self) -> List[Tuple[str, str, str]]:
        """Get a variety of test positions for comprehensive benchmarking"""
        return [
            ("Starting position", chess.STARTING_FEN, "Initial board setup"),
            ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "Early opening"),
            ("After 1.e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Open game"),
            ("After 1.e4 e5 2.Nf3", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Knight development"),
            ("Complex middlegame", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 4 4", "Active middlegame"),
            ("Endgame position", "8/8/8/8/8/8/4K3/4k3 w - - 0 1", "Simple endgame"),
        ]
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive speed benchmarks"""
        print("🏁 CHESS ENGINE SPEED BENCHMARK")
        print("=" * 50)
        
        # Test different depths (limited to depth 4 for speed)
        depths = [2, 3, 4]
        
        # Test different evaluator types
        evaluator_types = ["handcrafted"]
        
        for evaluator_type in evaluator_types:
            print(f"\n🔧 Testing {evaluator_type.upper()} evaluator")
            print("-" * 30)
            
            # Create engine with current evaluator (use config file depth)
            engine = MinimaxEngine(evaluator_type=evaluator_type)
            
            # Suppress engine output during benchmarks
            original_verbose = getattr(engine, 'verbose', False)
            if hasattr(engine, 'verbose'):
                engine.verbose = False
            
            try:
                # Test search speed at different depths
                self._benchmark_search_speeds(engine, depths)
                
                # Test evaluation speed
                self._benchmark_evaluation_speed(engine)
                
                # Test move generation speed
                self._benchmark_move_generation(engine)
                
                # Test position-specific performance
                self._benchmark_position_performance(engine, depths)  # Test all depths (2, 3, 4)
                
            finally:
                # Restore original verbose setting
                if hasattr(engine, 'verbose'):
                    engine.verbose = original_verbose
        
        # Print summary
        self._print_summary()
    
    def _benchmark_search_speeds(self, engine: MinimaxEngine, depths: List[int]):
        """Benchmark search speed at different depths"""
        print(f"🔍 Testing search speeds at depths {depths}...")
        
        # Use starting position for consistent testing
        board = chess.Board()
        
        for depth in depths:
            engine.depth = depth
            engine.nodes_searched = 0
            
            # Warm up
            _ = engine.get_move(board)
            
            # Actual benchmark
            start_time = time.perf_counter()
            move = engine.get_move(board)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            nodes_per_second = int(engine.nodes_searched / elapsed) if elapsed > 0 else 0
            
            print(f"  Depth {depth}: {elapsed:.3f}s, {engine.nodes_searched:,} nodes, {nodes_per_second:,} n/s")
            
            self.benchmark_results['search_speeds'].append({
                'depth': depth,
                'time': elapsed,
                'nodes': engine.nodes_searched,
                'nodes_per_second': nodes_per_second,
                'move': str(move)
            })
    
    def _benchmark_evaluation_speed(self, engine: MinimaxEngine):
        """Benchmark evaluation function speed"""
        print(f"📈 Testing evaluation speed...")
        
        total_evals = 0
        total_time = 0
        
        # Test evaluation on multiple positions
        for name, fen, description in self.test_positions[:4]:  # Test first 4 positions
            board = chess.Board(fen)
            
            # Warm up
            _ = engine.evaluation_manager.evaluator.evaluate(board)
            
            # Benchmark evaluation
            start_time = time.perf_counter()
            for _ in range(100):  # Multiple evaluations for accuracy
                _ = engine.evaluation_manager.evaluator.evaluate(board)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            evals_per_second = int(100 / elapsed) if elapsed > 0 else 0
            
            print(f"  {name}: {evals_per_second:,} evals/s")
            
            total_evals += 100
            total_time += elapsed
        
        avg_evals_per_second = int(total_evals / total_time) if total_time > 0 else 0
        print(f"  Average: {avg_evals_per_second:,} evals/s")
        
        self.benchmark_results['evaluation_speeds'].append({
            'total_evals': total_evals,
            'total_time': total_time,
            'evals_per_second': avg_evals_per_second
        })
    
    def _benchmark_move_generation(self, engine: MinimaxEngine):
        """Benchmark move generation speed"""
        print(f"🎯 Testing move generation speed...")
        
        total_moves = 0
        total_time = 0
        
        for name, fen, description in self.test_positions[:4]:
            board = chess.Board(fen)
            
            # Warm up
            _ = list(board.legal_moves)
            
            # Benchmark move generation
            start_time = time.perf_counter()
            for _ in range(1000):  # Multiple generations for accuracy
                moves = list(board.legal_moves)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            moves_per_ms = int(1000 / elapsed) if elapsed > 0 else 0
            
            print(f"  {name}: {moves_per_ms:,} moves/ms")
            
            total_moves += 1000
            total_time += elapsed
        
        avg_moves_per_ms = int(total_moves / total_time) if total_time > 0 else 0
        print(f"  Average: {avg_moves_per_ms:,} moves/ms")
        
        self.benchmark_results['move_generation_speeds'].append({
            'total_moves': total_moves,
            'total_time': total_time,
            'moves_per_ms': avg_moves_per_ms
        })
    
    def _benchmark_position_performance(self, engine: MinimaxEngine, depths: List[int]):
        """Benchmark performance across different positions"""
        print(f"🎮 Testing position-specific performance...")
        
        for name, fen, description in self.test_positions:
            board = chess.Board(fen)
            
            for depth in depths:
                test_engine = MinimaxEngine(depth=depth, evaluator_type="handcrafted")
                test_engine.nodes_searched = 0
                
                start_time = time.perf_counter()
                move = test_engine.get_move(board)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                nodes_per_second = int(test_engine.nodes_searched / elapsed) if elapsed > 0 else 0
                
                print(f"  {name} (depth {depth}): {elapsed:.3f}s, {test_engine.nodes_searched:,} nodes, {nodes_per_second:,} n/s")
                
                self.benchmark_results['position_performance'].append({
                    'position': name,
                    'depth': depth,
                    'time': elapsed,
                    'nodes': test_engine.nodes_searched,
                    'nodes_per_second': nodes_per_second,
                    'move': str(move)
                })
    
    def _print_summary(self):
        """Print comprehensive benchmark summary"""
        print("\n" + "=" * 50)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Calculate averages
        if self.benchmark_results['search_speeds']:
            search_speeds = [r['nodes_per_second'] for r in self.benchmark_results['search_speeds']]
            print(f"🔍 Average search speed: {statistics.mean(search_speeds):,} nodes/second")
        
        if self.benchmark_results['evaluation_speeds']:
            eval_speeds = [r['evals_per_second'] for r in self.benchmark_results['evaluation_speeds']]
            print(f"📈 Average evaluation speed: {statistics.mean(eval_speeds):,} evaluations/second")
        
        if self.benchmark_results['move_generation_speeds']:
            move_speeds = [r['moves_per_ms'] for r in self.benchmark_results['move_generation_speeds']]
            print(f"🎯 Average move generation: {statistics.mean(move_speeds):,} moves/millisecond")
        
        if self.benchmark_results['position_performance']:
            all_nodes = [r['nodes'] for r in self.benchmark_results['position_performance']]
            all_times = [r['time'] for r in self.benchmark_results['position_performance']]
            total_nodes = sum(all_nodes)
            total_time = sum(all_times)
            print(f"🎮 Total nodes searched: {total_nodes:,}")
            print(f"⏱️  Total benchmark time: {total_time:.2f} seconds")
        
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
        print("=" * 40)
        
        # Create engine
        engine = MinimaxEngine(evaluator_type="handcrafted")  # Use config file depth
        
        # Suppress engine output during benchmarks
        original_verbose = getattr(engine, 'verbose', False)
        if hasattr(engine, 'verbose'):
            engine.verbose = False
        
        try:
            # Quick search test
            board = chess.Board()
            engine.nodes_searched = 0
            
            start_time = time.perf_counter()
            move = engine.get_move(board)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            nodes_per_second = int(engine.nodes_searched / elapsed) if elapsed > 0 else 0
            
            print(f"🔍 Search test (depth 4):")
            print(f"  Time: {elapsed:.3f} seconds")
            print(f"  Nodes: {engine.nodes_searched:,}")
            print(f"  Speed: {nodes_per_second:,} nodes/second")
            print(f"  Move: {move}")
            
            # Quick evaluation test
            total_evals = 0
            total_time = 0
            
            for name, fen, description in self.test_positions[:3]:
                board = chess.Board(fen)
                
                start_time = time.perf_counter()
                for _ in range(50):  # Fewer evaluations for quick test
                    _ = engine.evaluation_manager.evaluator.evaluate(board)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                evals_per_second = int(50 / elapsed) if elapsed > 0 else 0
                
                print(f"📈 {name}: {evals_per_second:,} evals/s")
                
                total_evals += 50
                total_time += elapsed
            
            avg_evals_per_second = int(total_evals / total_time) if total_time > 0 else 0
            print(f"📈 Average evaluation: {avg_evals_per_second:,} evals/s")
            
            # Store results
            self.benchmark_results['search_speeds'].append({
                'depth': 4,
                'time': elapsed,
                'nodes': engine.nodes_searched,
                'nodes_per_second': nodes_per_second,
                'move': str(move)
            })
            
            self.benchmark_results['evaluation_speeds'].append({
                'total_evals': total_evals,
                'total_time': total_time,
                'evals_per_second': avg_evals_per_second
            })
            
        finally:
            # Restore original verbose setting
            if hasattr(engine, 'verbose'):
                engine.verbose = original_verbose

if __name__ == "__main__":
    tester = SpeedTester()
    tester.run_comprehensive_benchmark() 