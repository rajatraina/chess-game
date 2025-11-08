#!/usr/bin/env python3
"""
Comprehensive FEN position analysis script with enhanced logging options.

Usage:
    python3 scripts/test_search_from_fen.py "FEN_STRING" [options]

Options:
    --depth N          Maximum search depth (default: 6)
    --eval-depth N     Search depth for evaluating all legal moves (default: 2)
    --time N           Time budget in seconds (default: unlimited)
    --verbose          Enable detailed move evaluation logging
    --components       Show evaluation component breakdown
    --all-moves        Show evaluation for all legal moves
    --no-opening-book  Disable opening book lookup
    --quiet            Minimal output
    --help             Show this help message

Examples:
    python3 scripts/test_search_from_fen.py "r2qkb1r/pp3ppp/2n1bn2/4p1B1/3pN3/1N4P1/PPP1PPBP/R2Q1RK1 b kq - 1 10"
    python3 scripts/test_search_from_fen.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --verbose --all-moves
    python3 scripts/test_search_from_fen.py "FEN_STRING" --all-moves --eval-depth 4 --components
    python3 scripts/test_search_from_fen.py "8/8/8/8/8/8/8/8 w - - 0 1" --depth 8 --time 30
"""

import chess
import sys
import os
import argparse
import time
from typing import List, Dict, Any

# Add the chess_game directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chess_game'))

from engine import MinimaxEngine
import time

class MoveEvaluationCaptureEngine(MinimaxEngine):
    """Custom engine that captures move evaluations during search"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_evaluations = []  # Store move evaluations during search
        self.capture_evaluations = False
    
    def _iterative_deepening_search(self, board, start_time, time_budget, search_mode=None):
        """Override iterative deepening to capture move evaluations"""
        if self.capture_evaluations:
            return self._iterative_deepening_with_capture(board, start_time, time_budget, search_mode)
        else:
            return super()._iterative_deepening_search(board, start_time, time_budget, search_mode)
    
    def _iterative_deepening_with_capture(self, board, start_time, time_budget, search_mode=None):
        """Iterative deepening that captures move evaluations at each depth"""
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        self.move_evaluations = []
        
        # Evaluate each move with a shallow search to capture evaluations
        for move in legal_moves:
            board.push(move)
            # Use a shallow search to get a proper evaluation
            eval_score, _, _ = self._minimax(board, 2, -float('inf'), float('inf'), [], None, None)
            board.pop()
            
            self.move_evaluations.append((move, eval_score))
        
        # Sort by evaluation (best for current player first)
        if board.turn:  # White to move - higher is better
            self.move_evaluations.sort(key=lambda x: x[1], reverse=True)
        else:  # Black to move - lower is better
            self.move_evaluations.sort(key=lambda x: x[1])
        
        # Now run the normal iterative deepening search
        return super()._iterative_deepening_search(board, start_time, time_budget, search_mode)

class FENAnalyzer:
    """Enhanced FEN position analyzer with detailed logging options"""
    
    def __init__(self, verbose=False, show_components=False, show_all_moves=False, quiet=False):
        self.verbose = verbose
        self.show_components = show_components
        self.show_all_moves = show_all_moves
        self.quiet = quiet
        
    def analyze_position(self, fen: str, depth: int = 6, eval_depth: int = 2, time_budget: float = None, 
                        disable_opening_book: bool = False) -> Dict[str, Any]:
        """Analyze a FEN position with comprehensive logging"""
        
        if not self.quiet:
            print(f"🔍 Analyzing position: {fen}")
            print("=" * 80)
        
        # Create board from FEN
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}"}
        
        # Basic position info
        side_to_move = "White" if board.turn else "Black"
        legal_moves = list(board.legal_moves)
        
        if not self.quiet:
            print(f"📊 Position Info:")
            print(f"   Side to move: {side_to_move}")
            print(f"   Legal moves: {len(legal_moves)}")
            print(f"   Half-move clock: {board.halfmove_clock}")
            print(f"   Full-move number: {board.fullmove_number}")
            print(f"   Castling rights: {board.castling_rights}")
            print(f"   En passant: {board.ep_square}")
            print()
        
        # Create engine with enhanced logging
        engine = MoveEvaluationCaptureEngine(
            depth=depth,
            evaluator_type="handcrafted",
            quiet=self.quiet
        )
        
        # Disable opening book if requested
        if disable_opening_book:
            engine.opening_book = None
        
        # Show all moves evaluation if requested
        if self.show_all_moves and not self.quiet:
            self._analyze_all_moves(board, engine, eval_depth, time_budget)
        
        # Run the main search using the engine's proper flow
        if not self.quiet:
            print("🚀 Starting iterative deepening search...")
            print("=" * 80)
        
        start_time = time.time()
        best_move = engine.get_move(board, time_budget=time_budget, 
                                  disable_opening_book=disable_opening_book)
        search_time = time.time() - start_time
        
        # Collect results
        results = {
            "fen": fen,
            "side_to_move": side_to_move,
            "legal_moves_count": len(legal_moves),
            "best_move": best_move,
            "best_move_san": board.san(best_move),
            "search_time": search_time,
            "nodes_searched": engine.nodes_searched,
            "search_depth": getattr(engine, 'completed_depth', 'unknown'),
            "evaluation": getattr(engine, 'best_value', 'unknown')
        }
        
        # Get detailed evaluation if available
        if hasattr(engine, 'evaluation_manager') and hasattr(engine.evaluation_manager, 'evaluate_with_components'):
            try:
                components = engine.evaluation_manager.evaluate_with_components(board)
                results["evaluation_components"] = components
            except:
                pass
        
        # Show results
        if not self.quiet:
            self._display_results(results, board, engine)
        
        return results
    
    def _analyze_all_moves(self, board: chess.Board, engine: MoveEvaluationCaptureEngine, eval_depth: int = 2, time_budget: float = None):
        """Analyze and display evaluation for all legal moves using the engine's search flow"""
        print(f"📋 Evaluating all legal moves (using engine search flow with depth {eval_depth}):")
        print("-" * 70)
        
        # Enable evaluation capture
        engine.capture_evaluations = True
        
        # Temporarily set depth for move evaluation
        original_depth = engine.depth
        engine.depth = eval_depth
        
        # Use the engine's search flow to get move evaluations
        # This will call _iterative_deepening_with_capture which captures move evaluations
        try:
            # Use a reasonable fraction of the total time budget for move evaluation
            eval_time = min(10.0, time_budget * 0.1) if time_budget else 5.0
            print(f"ℹ️  Using {eval_time:.1f}s for move evaluation phase")
            best_move = engine.get_move(board, time_budget=eval_time, disable_opening_book=True)
            
            # Display the captured move evaluations
            if hasattr(engine, 'move_evaluations') and engine.move_evaluations:
                move_evals = []
                for move, eval_score in engine.move_evaluations:
                    # Get components if requested
                    if self.show_components and hasattr(engine.evaluation_manager, 'evaluate_with_components'):
                        try:
                            # Make the move to get components
                            board.push(move)
                            components = engine.evaluation_manager.evaluate_with_components(board)
                            board.pop()
                            eval_str = f"{eval_score:.2f} (M:{components['material']:.1f}, P:{components['positional']:.1f}, Mob:{components['mobility']:.1f}, KS:{components['king_safety']:.1f}, PS:{components['pawn_structure']:.1f})"
                        except:
                            eval_str = f"{eval_score:.2f}"
                    else:
                        eval_str = f"{eval_score:.2f}"
                    
                    move_evals.append((move, eval_score, eval_str))
                
                # Display results - show ALL moves when --all-moves is specified
                for i, (move, eval_score, eval_str) in enumerate(move_evals):
                    move_san = board.san(move)
                    print(f"   {i+1:2d}. {move_san:6s} ({move}) = {eval_str}")
            else:
                print("   No move evaluations captured")
                
        except Exception as e:
            print(f"   Error during move evaluation: {e}")
        
        # Restore original depth and disable capture
        engine.depth = original_depth
        engine.capture_evaluations = False
        
        print()
    
    def _display_results(self, results: Dict[str, Any], board: chess.Board, engine: MinimaxEngine):
        """Display comprehensive search results"""
        print("=" * 80)
        print("🎯 SEARCH RESULTS")
        print("=" * 80)
        
        # Best move
        print(f"🏆 Best move: {results['best_move_san']} ({results['best_move']})")
        
        # Evaluation
        if results['evaluation'] != 'unknown' and results['evaluation'] is not None:
            print(f"📊 Evaluation: {results['evaluation']:.2f}")
        
        # Show evaluation components if available
        if 'evaluation_components' in results:
            comp = results['evaluation_components']
            print(f"   Material: {comp['material']:.1f}")
            print(f"   Positional: {comp['positional']:.1f}")
            print(f"   Mobility: {comp['mobility']:.1f}")
            print(f"   King Safety: {comp['king_safety']:.1f}")
            print(f"   Pawn Structure: {comp['pawn_structure']:.1f}")
        
        # Search statistics
        print(f"⏱️  Search time: {results['search_time']:.2f}s")
        print(f"🔍 Nodes searched: {results['nodes_searched']:,}")
        if results['nodes_searched'] > 0:
            print(f"🚀 Search speed: {results['nodes_searched']/results['search_time']:.0f} nodes/s")
        
        if results['search_depth'] != 'unknown':
            print(f"📏 Search depth: {results['search_depth']} plies")
        
        # Transposition table stats
        if hasattr(engine, 'tt_hits') and hasattr(engine, 'tt_misses'):
            total_tt = engine.tt_hits + engine.tt_misses
            if total_tt > 0:
                print(f"💾 TT hit rate: {engine.tt_hits}/{total_tt} ({100*engine.tt_hits/total_tt:.1f}%)")
        
        # Show position after move
        board.push(results['best_move'])
        print(f"📍 Position after move: {board.fen()}")
        board.pop()
        
        # Show principal variation if available
        if hasattr(engine, 'best_line') and engine.best_line:
            pv_moves = []
            temp_board = board.copy()
            for move in engine.best_line[:5]:  # Show first 5 moves of PV
                if move in temp_board.legal_moves:
                    pv_moves.append(temp_board.san(move))
                    temp_board.push(move)
                else:
                    break
            if pv_moves:
                print(f"🎬 Principal variation: {' '.join(pv_moves)}")
        
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze chess positions from FEN strings with enhanced logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("fen", help="FEN string of the position to analyze")
    parser.add_argument("--depth", type=int, default=6, 
                       help="Maximum search depth (default: 6)")
    parser.add_argument("--eval-depth", type=int, default=2,
                       help="Search depth for evaluating all legal moves (default: 2)")
    parser.add_argument("--time", type=float, 
                       help="Time budget in seconds (default: unlimited)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable detailed move evaluation logging")
    parser.add_argument("--components", action="store_true",
                       help="Show evaluation component breakdown")
    parser.add_argument("--all-moves", action="store_true",
                       help="Show evaluation for all legal moves")
    parser.add_argument("--no-opening-book", action="store_true",
                       help="Disable opening book lookup")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FENAnalyzer(
        verbose=args.verbose,
        show_components=args.components,
        show_all_moves=args.all_moves,
        quiet=args.quiet
    )
    
    # Run analysis
    try:
        results = analyzer.analyze_position(
            fen=args.fen,
            depth=args.depth,
            eval_depth=args.eval_depth,
            time_budget=args.time,
            disable_opening_book=args.no_opening_book
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            sys.exit(1)
        
        # Exit with appropriate code
        if results['evaluation'] != 'unknown' and results['evaluation'] is not None:
            if results['evaluation'] > 0:
                sys.exit(0)  # White advantage
            elif results['evaluation'] < 0:
                sys.exit(2)  # Black advantage
            else:
                sys.exit(1)  # Equal position
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
