#!/usr/bin/env python3
"""
Unified logging manager for chess engine.
Provides consistent logging for both GUI and UCI interfaces.

IMPORTANT: NEVER use print() statements in the engine code when logging to lichess interface.
All logging must go through this logging manager to avoid polluting the UCI protocol.
The lichess interface expects clean UCI communication and any unexpected output will cause warnings.
"""

import time
import logging
from datetime import datetime

class ChessLoggingManager:
    """Unified logging manager for chess engine operations"""
    
    def __init__(self, log_callback=None, quiet=False, use_python_logging=False):
        """
        Initialize logging manager.
        
        Args:
            log_callback: Function to call for logging (default: print)
            quiet: If True, suppress all logging
            use_python_logging: If True, use Python logging instead of callback
        """
        self.log_callback = log_callback or print
        self.quiet = quiet
        self.search_start_time = None
        self.move_order_logged = False
        self.use_python_logging = use_python_logging
        
        if self.use_python_logging:
            self.logger = logging.getLogger(__name__)
    
    def log(self, message):
        """Log a message if not quiet"""
        if not self.quiet:
            if self.use_python_logging:
                # Use Python logging for lichess bot context
                self.logger.info(message)
            elif self.log_callback:
                # Use callback for other contexts (like pygame)
                self.log_callback(message)
    
    def log_search_start(self, board, search_depth):
        """Log the start of a search"""
        if not self.quiet:
            self.log(f"\n🤔 Engine thinking (depth {search_depth})...")
            self.log(f"🎭 Current side to move: {'White' if board.turn else 'Black'}")
            self.search_start_time = time.time()
            self.move_order_logged = False
    
    def log_iterative_deepening_start(self, starting_depth, max_depth):
        """Log the start of iterative deepening"""
        if not self.quiet:
            self.log(f"🔄 Iterative deepening: {starting_depth} → {max_depth} plies")
    
    def log_iteration_start(self, iteration, depth):
        """Log the start of a specific iteration"""
        if not self.quiet:
            self.log(f"🔄 Iteration {iteration}: depth {depth}")
    
    def log_iteration_move_order(self, board, moves, max_moves=8):
        """Log the move order for the current iteration"""
        if not self.quiet:
            try:
                # Convert moves to SAN notation
                move_sans = []
                for move in moves[:max_moves]:
                    try:
                        move_sans.append(board.san(move))
                    except Exception:
                        move_sans.append(move.uci())
                
                # Add ellipsis if there are more moves
                if len(moves) > max_moves:
                    move_sans.append("...")
                
                move_order_str = " ".join(move_sans)
                self.log(f"🎯 Move order: {move_order_str}")
            except Exception as e:
                self.log(f"🎯 Move order: Error generating move order: {e}")
    
    def log_iteration_complete(self, depth, iteration_time, best_move_san, best_value):
        """Log the completion of an iteration"""
        if not self.quiet:
            self.log(f"✅ Depth {depth} completed in {iteration_time:.2f}s: {best_move_san} ({best_value:.1f})")
    
    def log_iterative_deepening_timeout(self, current_depth, time_used, time_budget):
        """Log when iterative deepening times out"""
        if not self.quiet:
            self.log(f"⏰ Timeout at depth {current_depth} ({time_used:.2f}s/{time_budget:.2f}s)")
    
    def log_iterative_deepening_complete(self, final_depth, total_time, best_move_san, best_value):
        """Log the completion of iterative deepening"""
        if not self.quiet:
            self.log(f"🏁 Iterative deepening complete: depth {final_depth} in {total_time:.2f}s")
            self.log(f"🎯 Final best move: {best_move_san} ({best_value:.1f})")
    
    def log_move_order(self, board, ordered_moves, max_moves=5):
        """Log the move order from shallow search or other optimization"""
        if not self.quiet and not self.move_order_logged:
            try:
                # Convert moves to SAN notation
                move_sans = []
                for move in ordered_moves[:max_moves]:
                    try:
                        move_sans.append(board.san(move))
                    except Exception:
                        move_sans.append(move.uci())
                
                # Add ellipsis if there are more moves
                if len(ordered_moves) > max_moves:
                    move_sans.append("...")
                
                move_order_str = " ".join(move_sans)
                self.log(f"🎯 Move order: {move_order_str}")
                self.move_order_logged = True
            except Exception as e:
                self.log(f"🎯 Move order: Error generating move order: {e}")
    
    def log_new_best_move(self, move_san, evaluation):
        """Log a new best move found during search"""
        if not self.quiet:
            self.log(f"🔄 New best move: {move_san} ({evaluation:.1f})")
    
    def log_search_completion(self, search_time, nodes_searched, tt_stats):
        """Log search completion statistics"""
        if not self.quiet:
            nodes_per_second = nodes_searched / search_time if search_time > 0 else 0
            self.log(f"⏱️ Search completed in {search_time:.2f}s")
            if tt_stats and 'hits' in tt_stats:
                self.log(f"🔄 TT: {tt_stats['hits']}/{tt_stats['total_probes']} hits ({tt_stats['hit_rate']:.1f}%) | Cutoffs: {tt_stats['cutoffs']} ({tt_stats['cutoff_rate']:.1f}%)")
            self.log(f"🚀 Speed: {nodes_per_second:.0f} nodes/s")
    
    def log_best_move(self, board, best_move, best_value, best_line, eval_components=None):
        """Log the final best move with evaluation and principal variation"""
        if not self.quiet:
            try:
                move_san = board.san(best_move)
                
                # Generate principal variation
                pv_string = "N/A"
                if best_line:
                    try:
                        pv_moves = []
                        pv_board = board.copy()
                        for m in best_line[:5]:  # Show first 5 moves
                            if m in pv_board.legal_moves:
                                pv_moves.append(pv_board.san(m))
                                pv_board.push(m)
                            else:
                                break
                        pv_string = ' '.join(pv_moves) if pv_moves else "N/A"
                    except Exception as e:
                        pv_string = f"Error: {e}"
                
                # Log best move with evaluation and PV
                self.log(f"🏆 Best: {move_san} ({best_value:.1f}) | PV: {pv_string}")
                
                # Log evaluation components if available
                if eval_components:
                    material = eval_components.get('material', 0)
                    positional = eval_components.get('positional', 0)
                    mobility = eval_components.get('mobility', 0)
                    king_safety = eval_components.get('king_safety', 0)
                    overall_eval = material + positional + mobility + king_safety
                    self.log(f"📊 Overall: {overall_eval:.1f} (Material: {material:.1f}, Position: {positional:.1f}, Mobility: {mobility:.1f}, King Safety: {king_safety:.1f})")
                    
            except Exception as e:
                self.log(f"🏆 Best: {best_move.uci()} (SAN error: {e})")
    
    def log_shallow_search_stats(self, shallow_nodes, num_moves):
        """Log shallow search statistics"""
        if not self.quiet:
            self.log(f"📊 Shallow search completed: {shallow_nodes} nodes, {num_moves} moves evaluated")
    
    def log_tt_cleaning(self, entries_before, entries_after):
        """Log transposition table cleaning statistics"""
        if not self.quiet:
            entries_removed = entries_before - entries_after
            if entries_removed > 0:
                self.log(f"🧹 TT cleaned: removed {entries_removed} shallow entries ({entries_before} → {entries_after})")
    
    def log_time_budget(self, time_budget):
        """Log time budget allocation"""
        if not self.quiet:
            self.log(f"⏰ Time budget: {time_budget:.2f}s")
    
    def log_move_sent(self, move_san, search_completed):
        """Log the move that was sent"""
        if not self.quiet:
            status = "completed" if search_completed else "timeout"
            self.log(f"🎯 Move sent: {move_san} ({status})")
    
    def log_error(self, error_message):
        """Log an error message"""
        if not self.quiet:
            self.log(f"❌ Error: {error_message}")
    
    def log_info(self, info_message):
        """Log an info message"""
        if not self.quiet:
            self.log(f"ℹ️  {info_message}")
    
    def log_success(self, success_message):
        """Log a success message"""
        if not self.quiet:
            self.log(f"✅ {success_message}")
    
    def log_warning(self, warning_message):
        """Log a warning message"""
        if not self.quiet:
            self.log(f"⚠️  {warning_message}")
    
    def log_top_moves(self, board, move_evaluations, max_moves=5):
        """Log the top moves with their evaluations"""
        if not self.quiet and move_evaluations:
            try:
                # Sort moves by evaluation (best first for the side to move)
                # This assumes the moves are already sorted correctly from the search
                top_moves = move_evaluations[:max_moves]
                
                # Format the moves with their evaluations
                move_strings = []
                for move, evaluation in top_moves:
                    try:
                        # Get SAN notation for the move
                        move_san = board.san(move)
                        move_strings.append(f"{move_san} ({evaluation:.1f})")
                    except Exception:
                        # Fallback to UCI notation if SAN fails
                        move_strings.append(f"{move.uci()} ({evaluation:.1f})")
                
                top_moves_str = "  ".join(move_strings)
                self.log(f"Top moves: {top_moves_str}")
            except Exception as e:
                self.log(f"Top moves: Error formatting top moves: {e}")

# Global logging manager instance
_global_logger = None

def get_logger(log_callback=None, quiet=False, use_python_logging=False):
    """Get the global logging manager instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ChessLoggingManager(log_callback, quiet, use_python_logging)
    return _global_logger

def set_logger(logger):
    """Set the global logging manager instance"""
    global _global_logger
    _global_logger = logger
