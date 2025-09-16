#!/usr/bin/env python3
"""
Comprehensive FEN position analysis script with enhanced logging options.

Usage:
    python3 scripts/test_search_from_fen.py "FEN_STRING" [options]

Options:
    --depth N          Maximum search depth (default: 6)
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

class FENAnalyzer:
    """Enhanced FEN position analyzer with detailed logging options"""
    
    def __init__(self, verbose=False, show_components=False, show_all_moves=False, quiet=False):
        self.verbose = verbose
        self.show_components = show_components
        self.show_all_moves = show_all_moves
        self.quiet = quiet
        
    def analyze_position(self, fen: str, depth: int = 6, time_budget: float = None, 
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
        engine = MinimaxEngine(
            depth=depth,
            evaluator_type="handcrafted",
            quiet=self.quiet
        )
        
        # Disable opening book if requested
        if disable_opening_book:
            engine.opening_book = None
        
        # Show all moves evaluation if requested
        if self.show_all_moves and not self.quiet:
            self._analyze_all_moves(board, engine)
        
        # Run the main search
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
    
    def _analyze_all_moves(self, board: chess.Board, engine: MinimaxEngine):
        """Analyze and display evaluation for all legal moves"""
        print("📋 Evaluating all legal moves:")
        print("-" * 60)
        
        move_evals = []
        for move in board.legal_moves:
            # Make the move
            board.push(move)
            
            # Get evaluation
            if self.show_components and hasattr(engine.evaluation_manager, 'evaluate_with_components'):
                try:
                    components = engine.evaluation_manager.evaluate_with_components(board)
                    eval_score = components['total']
                    eval_str = f"{eval_score:.2f} (M:{components['material']:.1f}, P:{components['positional']:.1f}, Mob:{components['mobility']:.1f})"
                except:
                    eval_score = engine.evaluation_manager.evaluate(board)
                    eval_str = f"{eval_score:.2f}"
            else:
                eval_score = engine.evaluation_manager.evaluate(board)
                eval_str = f"{eval_score:.2f}"
            
            move_evals.append((move, eval_score, eval_str))
            
            # Undo the move
            board.pop()
        
        # Sort by evaluation (best for current player first)
        if board.turn:  # White to move - higher is better
            move_evals.sort(key=lambda x: x[1], reverse=True)
        else:  # Black to move - lower is better
            move_evals.sort(key=lambda x: x[1])
        
        # Display results
        for i, (move, eval_score, eval_str) in enumerate(move_evals[:10]):  # Show top 10
            move_san = board.san(move)
            print(f"   {i+1:2d}. {move_san:6s} ({move}) = {eval_str}")
        
        if len(move_evals) > 10:
            print(f"   ... and {len(move_evals) - 10} more moves")
        
        print()
    
    def _display_results(self, results: Dict[str, Any], board: chess.Board, engine: MinimaxEngine):
        """Display comprehensive search results"""
        print("=" * 80)
        print("🎯 SEARCH RESULTS")
        print("=" * 80)
        
        # Best move
        print(f"🏆 Best move: {results['best_move_san']} ({results['best_move']})")
        
        # Evaluation
        if results['evaluation'] != 'unknown':
            print(f"📊 Evaluation: {results['evaluation']:.2f}")
        
        # Show evaluation components if available
        if 'evaluation_components' in results:
            comp = results['evaluation_components']
            print(f"   Material: {comp['material']:.1f}")
            print(f"   Positional: {comp['positional']:.1f}")
            print(f"   Mobility: {comp['mobility']:.1f}")
        
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
            time_budget=args.time,
            disable_opening_book=args.no_opening_book
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            sys.exit(1)
        
        # Exit with appropriate code
        if results['evaluation'] != 'unknown':
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
