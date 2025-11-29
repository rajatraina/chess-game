import random
import chess
import chess.syzygy
import time
import os
import zlib
import math
from enum import Enum
try:
    from .evaluation import EvaluationManager, create_evaluator
    from .logging_manager import get_logger
    from .search_visualizer import get_visualizer, get_noop_visualizer, configure_visualizer
    from .opening_book import OpeningBook, OpeningBookError
except ImportError:
    from evaluation import EvaluationManager, create_evaluator
    from logging_manager import get_logger
    from search_visualizer import get_visualizer, get_noop_visualizer, configure_visualizer
    from opening_book import OpeningBook, OpeningBookError

# IMPORTANT: All logging must use self.logger methods, never print() statements.
# This ensures clean UCI protocol communication with lichess interface.


# Search status types
class SearchStatus(Enum):
    COMPLETE = "complete"      # Search completed successfully
    PARTIAL = "partial"        # Search timed out or was interrupted



class Engine:
    """Base class for chess engines"""
    def get_move(self, board):
        raise NotImplementedError

class RandomEngine(Engine):
    """Simple random move generator"""
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
        return None

class MinimaxEngine(Engine):
    """
    Chess engine using minimax search with alpha-beta pruning and quiescence search.
    
    Evaluation is always from White's perspective:
    - Positive values = good for White
    - Negative values = good for Black
    
    Search logic:
    - White's turn: maximize evaluation (find best move for White)
    - Black's turn: minimize evaluation (find best move for Black)
    """
    
    def __init__(self, depth=None, evaluator_type="handcrafted", evaluator_config=None, quiet=False, new_best_move_callback=None, use_python_logging=False):
        self.nodes_searched = 0
        self.search_start_time = 0
        self.quiet = quiet  # Control debug output
        self.new_best_move_callback = new_best_move_callback
        
        # Initialize logging manager
        self.logger = get_logger(new_best_move_callback, quiet, use_python_logging)
        
        # Initialize evaluation system with config file by default
        if evaluator_config is None:
            evaluator_config = {"config_file": "chess_game/evaluation_config.json"}
        
        self.evaluation_manager = EvaluationManager(evaluator_type, **(evaluator_config or {}))
        
        # Set search depth from config file if not explicitly provided
        if depth is None:
            self.depth = self.evaluation_manager.evaluator.config.get("search_depth", 4)
        else:
            self.depth = depth
        
        # Initialize Syzygy tablebase if available
        self.tablebase = None
        self.init_tablebase()
        
        
        # Search age for tracking
        self.search_age = 0
        
        # Killer moves storage - track cutoff counts per move per depth
        self.killer_moves = {}  # Dictionary: depth -> {move: cutoff_count}
        self.killer_moves_stored = 0
        self.killer_moves_used = 0
        
        
        # Load time management parameters
        self.time_management_moves_remaining = self.evaluation_manager.evaluator.config.get("time_management_moves_remaining", 30)
        self.time_management_moves_remaining_endgame = self.evaluation_manager.evaluator.config.get("time_management_moves_remaining_endgame", 10)
        self.time_management_increment_used = self.evaluation_manager.evaluator.config.get("time_management_increment_used", 0.8)
        
        # Load move ordering parameters
        self.moveorder_shallow_search_depth = self.evaluation_manager.evaluator.config.get("moveorder_shallow_search_depth", 2)
        
        # MVV-LVA piece values for capture move ordering (defined once at initialization)
        self.mvv_lva_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }
        
        # Load time budget parameters
        self.time_budget_check_frequency = self.evaluation_manager.evaluator.config.get("time_budget_check_frequency", 1000)
        self.time_budget_early_exit_enabled = self.evaluation_manager.evaluator.config.get("time_budget_early_exit_enabled", True)
        self.time_budget_safety_margin = self.evaluation_manager.evaluator.config.get("time_budget_safety_margin", 0.1)
        
        # Load quiescence parameters
        # Note: quiescence_additional_depth_limit is no longer used - max depth is derived from individual move type limits
        self.quiescence_additional_depth_limit_shallow_search = self.evaluation_manager.evaluator.config.get("quiescence_additional_depth_limit_shallow_search", 2)
        self.quiescence_additional_depth_limit_captures = self.evaluation_manager.evaluator.config.get("quiescence_additional_depth_limit_captures", 8)
        self.quiescence_additional_depth_limit_checks = self.evaluation_manager.evaluator.config.get("quiescence_additional_depth_limit_checks", 8)
        self.quiescence_additional_depth_limit_promotions = self.evaluation_manager.evaluator.config.get("quiescence_additional_depth_limit_promotions", 8)
        self.quiescence_additional_depth_limit_queen_defense = self.evaluation_manager.evaluator.config.get("quiescence_additional_depth_limit_queen_defense", 0)
        self.quiescence_additional_depth_limit_value_threshold = self.evaluation_manager.evaluator.config.get("quiescence_additional_depth_limit_value_threshold", 0)
        # Legacy flags (kept for backward compatibility, but now controlled by depth limits)
        self.quiescence_include_checks = self.evaluation_manager.evaluator.config.get("quiescence_include_checks", True)
        self.quiescence_include_queen_defense = self.evaluation_manager.evaluator.config.get("quiescence_include_queen_defense", False)
        self.quiescence_include_value_threshold = self.evaluation_manager.evaluator.config.get("quiescence_include_value_threshold", False)
        self.quiescence_value_threshold = self.evaluation_manager.evaluator.config.get("quiescence_value_threshold", 500)
        
        # Initialize search visualizer
        self.viz_enabled = self.evaluation_manager.evaluator.config.get("search_visualization_enabled", False)
        viz_target_fen = self.evaluation_manager.evaluator.config.get("search_visualization_target_fen", None)
        
        if self.viz_enabled:
            self.visualizer = get_visualizer()
            configure_visualizer(self.viz_enabled, viz_target_fen)
        else:
            self.visualizer = get_noop_visualizer()
        
        # Initialize opening book
        self.opening_book = None
        self.init_opening_book()
        
        # Move counter for visualization
        self.move_counter = 0
        
        # Initialize search result attributes
        self.best_value = None
        self.best_line = []
    
    def calculate_time_budget(self, time_remaining, increment, board=None):
        """
        Calculate time budget for the current move.
        
        Args:
            time_remaining: Time remaining in milliseconds
            increment: Time increment in milliseconds
            board: Current board state (optional, for endgame detection)
            
        Returns:
            Time budget in seconds
        """
        # Convert to seconds
        time_remaining_sec = time_remaining / 1000.0
        increment_sec = increment / 1000.0
        
        # Determine if this is an endgame position using existing evaluation logic
        is_endgame = False
        if board is not None:
            is_endgame = self.evaluation_manager.evaluator._is_endgame_evaluation()
        
        # Use appropriate moves remaining parameter
        moves_remaining = self.time_management_moves_remaining_endgame if is_endgame else self.time_management_moves_remaining
        
        # Calculate time budget using the formula
        time_budget = (time_remaining_sec / moves_remaining) + (increment_sec * self.time_management_increment_used)
        
        # Ensure minimum time budget of 0.1 seconds
        return max(time_budget, 0.1)
    
    def _shallow_search_with_quiescence(self, board, depth, alpha, beta, game_stage=None):
        """
        Perform a shallow search with limited quiescence for move ordering.
        
        Args:
            board: Current board state
            depth: Search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Tuple of (evaluation, principal_variation)
        """
        # Count this node
        self.nodes_searched += 1
        
        # Check for game over conditions
        if board.is_game_over():
            if board.is_checkmate():
                # Checkmate evaluation
                checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                if board.turn:  # Black wins (White to move but checkmated)
                    return -checkmate_bonus, [], SearchStatus.COMPLETE
                else:  # White wins (Black to move but checkmated)
                    return checkmate_bonus, [], SearchStatus.COMPLETE
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
                draw_value = self.evaluation_manager.evaluator.config.get("draw_value", 0)
                return draw_value, [], SearchStatus.COMPLETE  # Draw
            elif board.is_repetition():
                repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                return repetition_eval, [], SearchStatus.COMPLETE  # 3-fold repetition
        
        # Leaf node: use quiescence search
        if depth == 0:
            # Start quiescence depth at 0 (quiescence depth, not total depth)
            eval_score, line, status = self._quiescence_shallow(board, alpha, beta, 0, game_stage)
            return eval_score, line, status
        
        # Skip tablebase lookup for shallow search - we want speed, not perfect evaluation
        
        # Get sorted moves for this shallow search
        # Calculate absolute depth from root for killer move lookup
        absolute_depth = self.depth - depth

        sorted_moves = self._get_sorted_moves_optimized(board, absolute_depth)

        best_value = -float('inf') if board.turn else float('inf')
        best_line = []
        
        for move in sorted_moves:
            # Make the move
            board.push(move)
            
            # Search the resulting position
            value, line, status = self._shallow_search_with_quiescence(board, depth - 1, alpha, beta, game_stage)
            
            # If search was partial, propagate up immediately
            if status == SearchStatus.PARTIAL:
                board.pop()
                return None, [], SearchStatus.PARTIAL
            
            # Check for repetition while the move is still made
            repetition_after_move = board.is_repetition()
            
            # Undo the move
            board.pop()
            
            if repetition_after_move:
                repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                # Check current position evaluation to determine if we're winning/losing
                current_eval = self.evaluate(board)
                # If we're winning and this move leads to repetition, heavily penalize it
                # If we're losing and this move leads to repetition, prefer it
                if board.turn:  # White to move
                    if current_eval > 20:  # White is winning significantly
                        value = -1000  # Heavy penalty for throwing away winning position
                    elif current_eval < -20:  # White is losing significantly
                        value = repetition_eval  # Prefer draw over losing
                else:  # Black to move
                    if current_eval < -20:  # Black is winning significantly
                        value = 1000  # Heavy penalty for throwing away winning position
                    elif current_eval > 20:  # Black is losing significantly
                        value = repetition_eval  # Prefer draw over losing
            
            # Update best move based on whose turn it is
            if board.turn:  # White to move: maximize
                if value > best_value:
                    best_value = value
                    best_line = [move] + line
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cutoff
            else:  # Black to move: minimize
                if value < best_value:
                    best_value = value
                    best_line = [move] + line
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Alpha cutoff
        
        
        return best_value, best_line, SearchStatus.COMPLETE
    
    def _quiescence_shallow(self, board, alpha, beta, depth=0, game_stage=None):
        """
        Limited quiescence search for shallow move ordering.
        
        Args:
            board: Current board state
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Current quiescence depth
            
        Returns:
            Tuple of (evaluation, principal_variation, status)
        """
        # Count this node
        self.nodes_searched += 1
        
        # Check for game over conditions
        if board.is_game_over():
            return self.evaluate(board, game_stage), [], SearchStatus.COMPLETE
        
        # Limit quiescence depth
        # depth represents quiescence depth (starting at 0), limit is additional depth beyond base search
        max_quiescence_depth = self.quiescence_additional_depth_limit_shallow_search
        if depth > max_quiescence_depth:
            return self.evaluate(board, game_stage), [], SearchStatus.COMPLETE
        
        # Check if position is in check
        is_in_check = board.is_check()
        
        if is_in_check:
            # IN CHECK: Search ALL legal moves (must respond to check)
            moves_to_search = list(board.legal_moves)
            
            if not moves_to_search:
                # If in check and no legal moves, it's checkmate
                return -100000 if board.turn else 100000, [], SearchStatus.COMPLETE
        else:
            # NOT IN CHECK: Evaluate stand pat and search captures/checks
            stand_pat = self.evaluate(board, game_stage)
            
            # Alpha-beta pruning at quiescence level
            if board.turn:  # White to move: maximize
                if stand_pat >= beta:
                    return beta, [], SearchStatus.COMPLETE
                alpha = max(alpha, stand_pat)
            else:  # Black to move: minimize
                if stand_pat <= alpha:
                    return alpha, [], SearchStatus.COMPLETE
                beta = min(beta, stand_pat)
            
            # Search captures, checks, and promotions for tactical accuracy
            # For shallow search, keep it simple: just captures, checks, and promotions
            include_captures = True
            include_checks = True
            include_promotions = True
            include_queen_defense = False
            include_value_threshold = False
            
            moves_to_search = self._get_quiescence_moves(
                board,
                include_captures=include_captures,
                include_checks=include_checks,
                include_promotions=include_promotions,
                include_queen_defense=include_queen_defense,
                include_value_threshold=include_value_threshold
            )
            
            if not moves_to_search:
                # If not in check and no captures/checks, return stand pat
                return stand_pat, [], SearchStatus.COMPLETE
        
        best_line = []
        best_move = None
        
        for move in moves_to_search:
            # Make the move
            board.push(move)
            
            # Recursive quiescence search
            value, line, status = self._quiescence_shallow(board, alpha, beta, depth + 1, game_stage)
            
            # If search was partial, propagate up immediately
            if status == SearchStatus.PARTIAL:
                board.pop()
                return None, [], SearchStatus.PARTIAL
            
            # Undo the move
            board.pop()
            
            # Update best move
            if board.turn:  # White to move: maximize
                if value > alpha:
                    alpha = value
                    best_line = [move] + line
                    best_move = move
            else:  # Black to move: minimize
                if value < beta:
                    beta = value
                    best_line = [move] + line
                    best_move = move
            
            # Alpha-beta cutoff
            if alpha >= beta:
                break
        
        return (alpha if board.turn else beta), best_line, SearchStatus.COMPLETE
    
    def _order_moves_with_shallow_search(self, board, game_stage=None):
        """
        Order moves using shallow search with limited quiescence.
        
        Args:
            board: Current board state
            
        Returns:
            Tuple of (ordered_moves, shallow_search_stats) where shallow_search_stats contains
            nodes and moves count for predictive time management
        """
        if not self.quiet:
            self.logger.log_info(f"Performing shallow search (depth {self.moveorder_shallow_search_depth}) for move ordering...")
        
        # Store original node count
        original_nodes = self.nodes_searched
        
        # Perform shallow search on all legal moves
        move_evaluations = []
        legal_moves = list(board.legal_moves)
        
        for move in legal_moves:
            # Make the move
            board.push(move)
            
            # Perform shallow search
            eval_score, _, status = self._shallow_search_with_quiescence(
                board, 
                self.moveorder_shallow_search_depth - 1, 
                -float('inf'), 
                float('inf'),
                game_stage
            )
            
            # If search was partial, skip this move
            if status == SearchStatus.PARTIAL:
                board.pop()
                continue
            
            # Undo the move
            board.pop()
            
            move_evaluations.append((move, eval_score))
        
        # Sort moves by evaluation (best first for the side to move)
        if board.turn:  # White to move: sort by descending evaluation
            move_evaluations.sort(key=lambda x: x[1], reverse=True)
        else:  # Black to move: sort by ascending evaluation
            move_evaluations.sort(key=lambda x: x[1], reverse=False)
        
        # Extract ordered moves
        ordered_moves = [move for move, eval_score in move_evaluations]
        
        # Calculate shallow search statistics for predictive time management
        shallow_nodes = self.nodes_searched - original_nodes
        shallow_moves = len(legal_moves)
        # Use piece count from evaluator if available, otherwise fall back to board calculation
        if hasattr(self.evaluation_manager.evaluator, 'starting_piece_count') and self.evaluation_manager.evaluator.starting_piece_count is not None:
            num_pieces = self.evaluation_manager.evaluator.starting_piece_count
        else:
            num_pieces = chess.popcount(board.occupied)
        # Use overall search start time to show time since move calculation began
        total_elapsed_time = time.time() - self.search_start_time
        shallow_search_stats = {
            'nodes': shallow_nodes,
            'moves': shallow_moves,
            'num_pieces': num_pieces
        }
        
        # Log shallow search statistics and move order
        self.logger.log_shallow_search_stats(shallow_nodes, shallow_moves, total_elapsed_time)
        self.logger.log_move_order(board, ordered_moves)
        
        return ordered_moves, shallow_search_stats
    
    
    
    
    
    
    
    
    def init_tablebase(self):
        """Initialize Syzygy tablebase if available"""
        try:
            # Check local syzygy/3-4-5 directory
            tablebase_path = "./syzygy/3-4-5"
            
            if os.path.exists(tablebase_path):
                self.tablebase = chess.syzygy.open_tablebase(tablebase_path)
                return
            
            # No tablebases found - this is normal, no warning needed
            self.tablebase = None
            
        except Exception as e:
            if not self.quiet:
                self.logger.log_warning(f"Could not initialize tablebase: {e}")
            self.tablebase = None
    
    def init_opening_book(self):
        """Initialize opening book if enabled and available"""
        try:
            opening_book_config = self.evaluation_manager.evaluator.config.get("opening_book", {})
            
            if not opening_book_config.get("enabled", False):
                if not self.quiet:
                    self.logger.log_info("Opening book disabled in configuration")
                return
            
            book_file_path = opening_book_config.get("file_path", "opening_book.txt")
            
            # Try to load the opening book with time-based random seed
            time_seed = int(time.time() * 1000000)  # Use microseconds for better uniqueness
            self.opening_book = OpeningBook(book_file_path, random_seed=time_seed)
            
            # Configure opening book parameters from config
            eval_threshold = opening_book_config.get("eval_threshold", 0.1)
            min_games = opening_book_config.get("min_games", 10)
            
            self.opening_book.set_eval_threshold(eval_threshold)
            self.opening_book.set_min_games(min_games)
            
            if not self.quiet:
                stats = self.opening_book.get_book_stats()
                self.logger.log_info(f"Opening book loaded: {stats['total_positions']} positions, {stats['total_moves']} moves")
                self.logger.log_info(f"Opening book config: eval threshold={eval_threshold}, min games={min_games}")
            
        except OpeningBookError as e:
            if not self.quiet:
                self.logger.log_error(f"Opening book error: {e}")
            self.opening_book = None
        except Exception as e:
            if not self.quiet:
                self.logger.log_warning(f"Could not initialize opening book: {e}")
            self.opening_book = None

    def get_move(self, board, time_budget=None, repetition_detected=False, disable_opening_book=False):
        """
        Find the best move for the current side to move.
        
        Args:
            board: Current chess board state
            time_budget: Maximum time to spend on this move in seconds (None for unlimited)
            repetition_detected: Whether the current position is a 3-fold repetition
            
        Returns:
            Best move found by the search
        """
        # Increment search age
        self.search_age += 1
        
        # Clear killer moves for new search to avoid stale data
        self._clear_killer_moves()
        
        # Set the starting position for consistent evaluation throughout search
        if hasattr(self.evaluation_manager.evaluator, '_set_starting_position'):
            self.evaluation_manager.evaluator._set_starting_position(board)
            if not self.quiet:
                self.logger.log_info(f"Set starting position: {chess.popcount(board.occupied)} pieces")
        else:
            # Fallback: try to set starting position directly if the evaluator has the attribute
            if hasattr(self.evaluation_manager.evaluator, 'starting_piece_count'):
                self.evaluation_manager.evaluator.starting_piece_count = chess.popcount(board.occupied)
                if not self.quiet:
                    self.logger.log_info(f"Set starting position (fallback): {chess.popcount(board.occupied)} pieces")
            else:
                if not self.quiet:
                    self.logger.log_warning("Could not set starting position - evaluator missing required attributes")
        
        # Clear evaluation caches based on game stage (only clears if stage changed or cache too large)
        if hasattr(self.evaluation_manager.evaluator, 'clear_eval_cache'):
            # Determine game stage for cache management
            if hasattr(self.evaluation_manager.evaluator, '_determine_game_stage'):
                game_stage = self.evaluation_manager.evaluator._determine_game_stage(board)
                self.evaluation_manager.evaluator.clear_eval_cache(game_stage)
            else:
                # Fallback: clear unconditionally if game stage determination not available
                self.evaluation_manager.evaluator.clear_eval_cache()
        
        # Check if position is in tablebase
        if self.tablebase and self.is_tablebase_position(board):
            if not self.quiet:
                piece_count = sum(len(board.pieces(piece_type, color)) for piece_type in chess.PIECE_TYPES for color in [chess.WHITE, chess.BLACK])
                self.logger.log_info(f"Checking tablebase for position with {piece_count} pieces")
            tablebase_move = self.get_tablebase_move(board)
            if tablebase_move:
                if not self.quiet:
                    self.logger.log_info(f"Using tablebase move: {board.san(tablebase_move)}")
                return tablebase_move
            else:
                if not self.quiet:
                    self.logger.log_warning("Tablebase lookup failed, using standard search")
        
        # Check if position is in opening book
        if self.opening_book and not disable_opening_book and self.opening_book.is_in_book(board):
            available_moves_detailed = self.opening_book.get_available_moves_detailed(board)
            opening_move = self.opening_book.get_move(board)
            if opening_move:
                # Format: ℹ️  Opening Book hit. Found moves: e6 [0.625] d4 [0.622]. Playing move: e6
                moves_str = " ".join([f"{move} [{eval:.3f}]" for move, count, eval in available_moves_detailed])
                message = f"ℹ️  Opening Book hit. Found moves: {moves_str}. Playing move: {board.san(opening_move)}"
                self.logger.log_info(message)
                return opening_move
            else:
                self.logger.log_warning("Opening book lookup failed, using standard search")
        
        # Start timing the search and reset node counter
        start_time = time.time()
        self.search_start_time = start_time
        self.time_budget = time_budget  # Store time budget for minimax function
        self.nodes_searched = 0
        self.search_interrupted = False  # Track if search was interrupted by time budget
        self.repetition_detected = repetition_detected  # Store repetition information
        
        # Use unified iterative deepening search for all cases
        min_time_budget = self.evaluation_manager.evaluator.config.get("search_deepening_min_time_budget", 30.0)
        
        if time_budget is None:
            # No time budget: use iterative deepening with no time limit (for pygame)
            self.logger.log_info("No time budget provided, using iterative deepening without time limit")
            return self._iterative_deepening_search(board, start_time, None, None)
        elif time_budget < min_time_budget:
            # Time budget too small: use iterative deepening with limited depth (for lichess with very short time)
            self.logger.log_info(f"Time budget {time_budget:.1f}s below minimum {min_time_budget}s, using limited iterative deepening")
            return self._iterative_deepening_search(board, start_time, time_budget, "limited")
        else:
            # Sufficient time budget: use full iterative deepening (for lichess)
            self.logger.log_info(f"Time budget {time_budget:.1f}s sufficient for full iterative deepening (min: {min_time_budget}s)")
            return self._iterative_deepening_search(board, start_time, time_budget, "full")



    def _iterative_deepening_search(self, board, start_time, time_budget, search_mode=None):
        """
        Perform iterative deepening search (for lichess interface).
        
        Args:
            board: Current board state
            start_time: Search start time
            time_budget: Time budget in seconds
            search_mode: "limited", "full", or None for unlimited
            
        Returns:
            Best move found
        """
        # Determine game stage once at root level for consistent evaluation throughout search
        if hasattr(self.evaluation_manager.evaluator, '_determine_game_stage'):
            game_stage = self.evaluation_manager.evaluator._determine_game_stage(board)
        else:
            game_stage = None  # Will be determined automatically in evaluate calls
        
        # Get configuration parameters
        starting_depth = self.evaluation_manager.evaluator.config.get("search_depth_starting", 3)
        max_depth = self.evaluation_manager.evaluator.config.get("max_search_depth", 10)
        time_fraction = self.evaluation_manager.evaluator.config.get("search_deepening_time_budget_fraction", 0.8)
        
        # Limit max_depth for limited search mode
        if search_mode == "limited":
            max_depth = starting_depth
            self.logger.log_info(f"Limited search mode: max_depth set to {max_depth}")
        
        
        # Initialize best move tracking
        best_move = None
        best_value = -float('inf') if board.turn else float('inf')
        best_line = []
        previous_iteration_best_moves = []  # Track best moves from previous iteration for ordering
        previous_iteration_move_order = []  # Track the complete move order from previous iteration
        
        # Phase 1: Perform shallow search for initial move ordering
        self.logger.log_info("Phase 1: Shallow search (depth 2) for move ordering...")
        sorted_moves, shallow_search_stats = self._order_moves_with_shallow_search(board, game_stage)
        
        
        # Phase 2: Determine optimal starting depth using predictive time management
        if time_budget is not None and self.evaluation_manager.evaluator.config.get("predictive_time_management_enabled", True):
            # Calculate remaining time after shallow search and other processing
            elapsed_time = time.time() - start_time
            remaining_time = time_budget - elapsed_time
            
            # Use predictive time management to determine optimal starting depth
            optimal_starting_depth = self._predict_optimal_starting_depth(shallow_search_stats, remaining_time, starting_depth, game_stage)
            self.logger.log_info(f"Predictive time management: elapsed={elapsed_time:.2f}s, remaining={remaining_time:.2f}s, using depth {optimal_starting_depth} as starting depth")
        else:
            # Use default starting depth
            optimal_starting_depth = starting_depth
            self.logger.log_info(f"Using default starting depth: {optimal_starting_depth}")
        
        # Phase 3: Iterative deepening loop
        current_depth = optimal_starting_depth
        completed_depth = None  # Track the depth that was actually completed
        iteration = 1
        
        # Calculate current move number (1-based)
        current_move_number = len(board.move_stack) // 2 + 1
        
        # Log iterative deepening start
        self.logger.log_iterative_deepening_start(optimal_starting_depth, max_depth)
        
        while current_depth <= max_depth:
            # Log iteration start
            self.logger.log_iteration_start(iteration, current_depth)
        
            
            # Initialize for this iteration
            alpha = -float('inf')
            beta = float('inf')
            iteration_has_timeouts = False  # Track if any moves timed out
            first_completed_move = None  # Track if we got at least one completed move
            move_evaluations = []  # Track move evaluations for logging only
            moves_that_were_best = []  # Track all moves that were best at any point during this iteration
            
            # Start search visualization for this iteration
            self.visualizer.start_search(board, current_depth)
            
            # Evaluate moves in optimal order
            if iteration > 1 and previous_iteration_best_moves and previous_iteration_move_order:
                # Use reverse order of best moves from previous iteration
                # Best move from previous iteration should be searched first
                reordered_moves = self._create_move_order_from_previous_best(
                    board, previous_iteration_best_moves, previous_iteration_move_order
                )
            else:
                # First iteration uses shallow search order
                reordered_moves = sorted_moves
            
            # Log move order for this iteration
            self.logger.log_iteration_move_order(board, reordered_moves)
            
            for move in reordered_moves:
                # Check time budget (only if time_budget is provided)
                if time_budget is not None:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= time_budget:
                        self.logger.log_iterative_deepening_timeout(current_depth, elapsed_time, time_budget)
                        self.search_interrupted = True
                        break
                
                # Record move being considered for visualization
                self.visualizer.record_move_considered(move, board)
                
                # Get the SAN notation before making the move
                move_san = board.san(move)
                
                # Store the original turn before making the move
                original_turn = board.turn
                
                # Make the move on the board
                board.push(move)
                
                # Search the resulting position
                value, line, status = self._minimax(board, current_depth - 1, alpha, beta, [move_san], start_time, time_budget, game_stage)
                
                if status == SearchStatus.PARTIAL:
                    # Undo the move to restore the original board state
                    board.pop()
                    # Skip this move due to timeout - log and stop searching remaining moves
                    self.logger.log_info(f"Move {move_san} timed out during search")
                    iteration_has_timeouts = True  # Mark that this iteration had timeouts
                    break
                
                # Move completed successfully
                # Undo the move to restore the original board state
                board.pop()

                # Update global best move - compare against existing best move
                if first_completed_move is None:
                    # This is the first completed move at this depth - it's automatically the best
                    best_move = move
                    best_value = value
                    best_line = [move] + line
                    first_completed_move = move
                    # Update instance variables for logging
                    self.best_line = best_line
                    self.best_value = best_value
                    # Log the new best move
                    self.logger.log_new_best_move(board.san(move), value)
                    self.logger.log_info(f"Updated best move from depth {current_depth}: {board.san(move)} ({value:.1f})")
                    # Track this as a move that was best
                    moves_that_were_best.append(move)
                else:
                    # Compare this move against the existing best move
                    is_better = False
                    if original_turn:  # White to move: maximize (higher is better)
                        is_better = value > best_value
                    else:  # Black to move: minimize (lower is better)
                        is_better = value < best_value
                    
                    if is_better:
                        # This move is better - update best move
                        best_move = move
                        best_value = value
                        best_line = [move] + line
                        # Update instance variables for logging
                        self.best_line = best_line
                        self.best_value = best_value
                        # Log the new best move
                        self.logger.log_new_best_move(board.san(move), value)
                        self.logger.log_info(f"Updated best move from depth {current_depth}: {board.san(move)} ({value:.1f})")
                        # Track this as a move that was best
                        moves_that_were_best.append(move)
                
                # Update alpha-beta bounds
                if original_turn:  # White to move: maximize
                    alpha = max(alpha, value)
                else:  # Black to move: minimize
                    beta = min(beta, value)
                
                # Track move evaluation for logging
                move_evaluations.append((move, value))
            
            # Update completed_depth based on whether all moves completed
            if first_completed_move is not None and not iteration_has_timeouts:
                # All moves completed - mark this depth as completed
                completed_depth = current_depth
                self.logger.log_info(f"Depth {current_depth} completed successfully")
            elif iteration_has_timeouts:
                self.logger.log_info(f"Depth {current_depth} had timeouts - using best move from incomplete depth {current_depth}")
            else:
                self.logger.log_info(f"Depth {current_depth} had no completed moves - keeping previous best move from depth {completed_depth}")
            
            # Store all moves that were best at any point during this iteration
            if moves_that_were_best:
                # Store them in reverse order (most recent best move first)
                # This creates the move order for the next iteration
                previous_iteration_best_moves = list(reversed(moves_that_were_best))
            
            # Store the complete move order from this iteration for next iteration
            previous_iteration_move_order = reordered_moves
            
            # Log iteration completion
            iteration_time = time.time() - start_time
            
            # Get best move SAN for logging
            best_move_san = "N/A"
            if best_move:
                try:
                    best_move_san = board.san(best_move)
                except Exception:
                    best_move_san = best_move.uci()
            
            self.logger.log_iteration_complete(current_depth, iteration_time, best_move_san, best_value)
            
            # Log top moves for this iteration
            if move_evaluations:
                self.logger.log_top_moves(board, move_evaluations, max_moves=5)
            
            # Check if we can go deeper (only if time_budget is provided)
            if time_budget is not None and iteration_time >= time_budget * time_fraction:
                self.logger.log_iterative_deepening_timeout(current_depth, iteration_time, time_budget)
                break
            
            # Check if there's only one legal move - no need to go deeper
            if len(reordered_moves) == 1:
                self.logger.log_info(f"Only one legal move available - stopping iterative deepening")
                break
            
            # Check if mate or mate in 2 was found (any iteration)
            if first_completed_move is not None and best_move is not None:
                is_mate, distance_to_mate = self._is_mate_found(best_value, best_line)
                if is_mate:
                    best_move_san = "N/A"
                    try:
                        best_move_san = board.san(best_move)
                    except Exception:
                        best_move_san = best_move.uci()
                    if distance_to_mate == 0:
                        self.logger.log_info(f"Checkmate found: {best_move_san} - stopping iterative deepening")
                    else:
                        self.logger.log_info(f"Mate in {distance_to_mate + 1} found: {best_move_san} - stopping iterative deepening")
                    break
            
            # Check if clear best capture was found (after first iteration)
            if iteration == 1 and first_completed_move is not None and best_move is not None:
                if self._is_clear_best_capture(board, best_move, sorted_moves[0]):
                    best_move_san = "N/A"
                    try:
                        best_move_san = board.san(best_move)
                    except Exception:
                        best_move_san = best_move.uci()
                    self.logger.log_info(f"Clear best capture detected: {best_move_san} - stopping iterative deepening")
                    break
            
            # Prepare for next iteration
            current_depth += 2  # Increment by 2 plies to keep same side to move
            iteration += 1
        
        # Log final search completion
        total_time = time.time() - start_time
        self.logger.log_search_completion(total_time, self.nodes_searched, {})
        
        # Log iterative deepening completion
        best_move_san = "N/A"
        if best_move:
            try:
                best_move_san = board.san(best_move)
            except Exception:
                best_move_san = best_move.uci()
        
        self.logger.log_iterative_deepening_complete(completed_depth, total_time, best_move_san, best_value)
        
        # Safety check: if no best move found, use first move from shallow search
        if best_move is None:
            if sorted_moves:
                best_move = sorted_moves[0]
                self.logger.log_warning(f"No best move found, using first move from shallow search: {board.san(best_move)}")
            else:
                self.logger.log_error("No legal moves available!")
        
        # Finish search visualization and logging
        self._finish_search_logging(board, best_move, best_value, best_line, total_time)
        
        return best_move

    def _create_move_order_from_previous_best(self, board, previous_best_moves, previous_move_order):
        """
        Create move order for current iteration based on all moves that were best in previous iteration.
        
        The strategy is to search all moves that were best in the previous iteration first,
        in reverse order (most recent best move first), then use the previous iteration's
        move order for any remaining moves.
        
        Args:
            board: Current board position
            previous_best_moves: List of moves that were best at any point in previous iteration (in reverse order)
            previous_move_order: Complete move order from previous iteration
            
        Returns:
            List of moves in optimal order for current iteration
        """
        # Start with all moves that were best in previous iteration (already in reverse order)
        ordered_moves = list(previous_best_moves)
        
        # Add all moves from previous iteration's order that weren't in the previous best moves
        # This ensures we don't miss any legal moves and maintains the previous iteration's ordering
        for move in previous_move_order:
            if move not in ordered_moves:
                ordered_moves.append(move)
        
        return ordered_moves

    def _finish_search_logging(self, board, best_move, best_value, best_line, search_time):
        """
        Finish search logging and visualization.
        
        Args:
            board: Current board state
            best_move: Best move found
            best_value: Best evaluation value
            best_line: Principal variation
            search_time: Total search time
        """
        if not best_move:
            return
        
        # Finish search visualization if enabled
        if self.viz_enabled:
            pv_san = []
            pv_board = board.copy()
            for m in best_line:
                try:
                    pv_san.append(pv_board.san(m))
                    pv_board.push(m)
                except Exception:
                    break
            self.visualizer.finish_search(board.san(best_move), best_value, self.nodes_searched, search_time, pv_san)
        else:
            self.visualizer.finish_search("", best_value, self.nodes_searched, search_time, [])
        
        # Export visualization to file
        self.move_counter += 1
        viz_file = self.visualizer.export_tree_to_file(self.move_counter)
        if viz_file:
            self.logger.log_info(f"Search tree exported to: {viz_file}")
        
        # Print the best move found with component evaluation
        pv_board = board.copy()
        pv_san = []
        for m in best_line:
            try:
                pv_san.append(pv_board.san(m))
                pv_board.push(m)
            except Exception:
                break
        
        # Compute evaluation components for the position at the end of the principal variation
        pv_board = board.copy()
        for move in best_line:
            pv_board.push(move)
        final_components = self.evaluate_with_components(pv_board)
        
        self.logger.log_best_move(board, best_move, best_value, best_line, final_components)

    def _minimax(self, board, depth, alpha, beta, variation=None, start_time=None, time_budget=None, game_stage=None):
        """
        Minimax search with alpha-beta pruning with transposition table support.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning (best score for maximizing player)
            beta: Beta value for pruning (best score for minimizing player)
            variation: The sequence of moves that led to this position (for debugging)
            start_time: Start time for time budget checking
            time_budget: Time budget in seconds for early exit
            
        Returns:
            Tuple of (evaluation, principal_variation, status)
            - evaluation: The evaluation value (None if status is PARTIAL)
            - principal_variation: The best line (empty list if status is PARTIAL)
            - status: SearchStatus.COMPLETE or SearchStatus.PARTIAL
        """

        # Enter node for visualization
        self.visualizer.enter_node(board, None, depth, alpha, beta, False)
        
        # Count this node
        self.nodes_searched += 1
        
        # Check time budget if enabled
        if (start_time and time_budget and self.time_budget_early_exit_enabled):
            
            # Check frequency to avoid performance impact
            if self.nodes_searched % self.time_budget_check_frequency == 0:
                elapsed_time = time.time() - start_time
                
                if elapsed_time >= time_budget - self.time_budget_safety_margin:
                    # Return partial status - search timed out
                    self.visualizer.exit_node(None, "TIMEOUT", [], 0, 0)
                    return None, [], SearchStatus.PARTIAL
        
        # Initialize variation if None
        if variation is None:
            variation = []
                
        # Leaf node: evaluate position
        if depth == 0:
            # Check if position is in tablebase for perfect evaluation
            if self.tablebase and self.is_tablebase_position(board):
                try:
                    wdl = self.tablebase.get_wdl(board)
                    if wdl is not None:
                        # Convert WDL to evaluation score
                        # WDL values are from the perspective of the side to move
                        # WDL: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
                        if board.turn:  # White to move
                            if wdl == 2:  # White wins
                                return 100000, [], SearchStatus.COMPLETE
                            elif wdl == 1:  # White cursed win
                                return 50000, [], SearchStatus.COMPLETE
                            elif wdl == 0:  # Draw
                                return 0, [], SearchStatus.COMPLETE
                            elif wdl == -1:  # White blessed loss
                                return -50000, [], SearchStatus.COMPLETE
                            elif wdl == -2:  # White loses
                                return -100000, [], SearchStatus.COMPLETE
                        else:  # Black to move
                            if wdl == 2:  # Black wins (bad for White)
                                return -100000, [], SearchStatus.COMPLETE
                            elif wdl == 1:  # Black cursed win (bad for White)
                                return -50000, [], SearchStatus.COMPLETE
                            elif wdl == 0:  # Draw
                                return 0, [], SearchStatus.COMPLETE
                            elif wdl == -1:  # Black blessed loss (good for White)
                                return 50000, [], SearchStatus.COMPLETE
                            elif wdl == -2:  # Black loses (good for White)
                                return 100000, [], SearchStatus.COMPLETE
                except Exception:
                    # Fall back to quiescence if tablebase lookup fails
                    pass
            
            # Enter quiescence node for visualization
            self.visualizer.enter_node(board, None, 0, alpha, beta, True)
            
            # Perform quiescence search
            # Start quiescence depth at 0 (quiescence depth, not total depth)
            eval_score, line, status = self._quiescence(board, alpha, beta, 0, game_stage)
            
            # Exit quiescence node for visualization
            self.visualizer.exit_node(eval_score, "QUIESCENCE", line if line else [], 0, 0)
            
            return eval_score, line, status
        
        # Base case: game over
        if board.is_game_over():
            if board.is_checkmate():
                # Checkmate evaluation
                checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                if board.turn:  # Black wins (White to move but checkmated)
                    eval_score = -checkmate_bonus
                else:  # White wins (Black to move but checkmated)
                    eval_score = checkmate_bonus
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
                # Draw conditions (except repetition)
                eval_score = self.evaluation_manager.evaluator.config.get("draw_value", 0)
            elif board.is_repetition():
                # 3-fold repetition - can be evaluated differently
                repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                eval_score = repetition_eval
            else:
                # Other game over conditions
                eval_score = self.evaluate(board, game_stage)
            
            # Exit node for visualization
            self.visualizer.exit_node(eval_score, "TERMINAL", [], 0, 0)
            return eval_score, [], SearchStatus.COMPLETE
        
        # Use optimized move generation and sorting
        # Calculate absolute depth from root for killer move lookup
        # In iterative deepening, the depth parameter is the remaining depth
        # So absolute_depth = max_search_depth - depth
        absolute_depth = self.depth - depth
        sorted_moves = self._get_sorted_moves_optimized(board, absolute_depth)
        
        # White's turn: maximize evaluation
        if board.turn:
            max_eval = -float('inf')
            best_line = []
            best_move = None
            original_alpha = alpha
            
            for move in sorted_moves:
                # Record move being considered for visualization
                self.visualizer.record_move_considered(move, board)
                
                # Get SAN notation before making the move
                move_san = board.san(move)
                
                # Check time budget before processing this move
                if (start_time and time_budget and self.time_budget_early_exit_enabled):
                    elapsed_time = time.time() - start_time
                    
                    if elapsed_time >= time_budget - self.time_budget_safety_margin:
                        # Return partial status - search timed out
                        self.visualizer.exit_node(None, "TIMEOUT", [], 0, 0)
                        return None, [], SearchStatus.PARTIAL
                
                # Make move
                board.push(move)
                # Recursively search the resulting position
                eval, line, status = self._minimax(board, depth - 1, alpha, beta, variation + [move_san], start_time, time_budget, game_stage)
                
                # If search was partial, propagate up immediately
                if status == SearchStatus.PARTIAL:
                    board.pop()
                    self.visualizer.exit_node(None, "PARTIAL", [], 0, 0)
                    return None, [], SearchStatus.PARTIAL
                # Undo move
                board.pop()
                
                # Checkmate distance correction
                checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                if abs(abs(eval) - checkmate_bonus) <= 1000:
                    if eval > 0:  # White is winning
                        distance_to_mate = len(line)
                        eval = checkmate_bonus - distance_to_mate
                    else:  # Black is winning
                        distance_to_mate = len(line)
                        eval = -(checkmate_bonus - distance_to_mate)
                
                # Check if this move leads to a 3-fold repetition
                board.push(move)
                repetition_after_move = board.is_repetition()
                board.pop()
                
                # Apply repetition penalty/bonus based on whether this move leads to repetition
                if repetition_after_move:
                    repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                    # Check current position evaluation to determine if we're winning/losing
                    current_eval = self.evaluate(board)
                    # If we're winning and this move leads to repetition, heavily penalize it
                    # If we're losing and this move leads to repetition, prefer it
                    if current_eval > 20:  # White is winning significantly
                        eval = -1000  # Heavy penalty for throwing away winning position
                    elif current_eval < -20:  # White is losing significantly
                        eval = repetition_eval  # Prefer draw over losing
                

                
                # Update best move if this is better
                if eval > max_eval:
                    max_eval = eval
                    best_line = [move] + line
                    best_move = move
                
                # Alpha-beta pruning
                alpha = max(alpha, eval)
                if beta <= alpha:
                    # Store killer move before breaking
                    self._store_killer_move(move, absolute_depth, board)
                    break  # Beta cutoff
            
            # Exit node for visualization
            if self.viz_enabled:
                best_line_san = []
                if best_line:
                    temp_board = board.copy()
                    for move in best_line:
                        try:
                            best_line_san.append(temp_board.san(move))
                            temp_board.push(move)
                        except Exception:
                            best_line_san.append(move.uci())
                self.visualizer.exit_node(max_eval, "EXACT", best_line_san, 0, 0)
            else:
                self.visualizer.exit_node(max_eval, "EXACT", [], 0, 0)
            return max_eval, best_line, SearchStatus.COMPLETE
            
        # Black's turn: minimize evaluation
        else:
            min_eval = float('inf')
            best_line = []
            best_move = None
            original_beta = beta
            
            for move in sorted_moves:
                # Record move being considered for visualization
                self.visualizer.record_move_considered(move, board)
                
                # Get SAN notation before making the move
                move_san = board.san(move)
                
                # Check time budget before processing this move
                if (start_time and time_budget and self.time_budget_early_exit_enabled):
                    elapsed_time = time.time() - start_time
                    
                    if elapsed_time >= time_budget - self.time_budget_safety_margin:
                        # Return partial status - search timed out
                        self.visualizer.exit_node(None, "TIMEOUT", [], 0, 0)
                        return None, [], SearchStatus.PARTIAL
                
                # Make move
                board.push(move)
                # Recursively search the resulting position
                eval, line, status = self._minimax(board, depth - 1, alpha, beta, variation + [move_san], start_time, time_budget, game_stage)
                
                # If search was partial, propagate up immediately
                if status == SearchStatus.PARTIAL:
                    board.pop()
                    self.visualizer.exit_node(None, "PARTIAL", [], 0, 0)
                    return None, [], SearchStatus.PARTIAL
                # Undo move
                board.pop()
                
                # Checkmate distance correction
                checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                if abs(abs(eval) - checkmate_bonus) <= 1000:
                    if eval > 0:  # White is winning
                        distance_to_mate = len(line)
                        eval = checkmate_bonus - distance_to_mate
                    else:  # Black is winning
                        distance_to_mate = len(line)
                        eval = -(checkmate_bonus - distance_to_mate)
                
                # Check if this move leads to a 3-fold repetition
                board.push(move)
                repetition_after_move = board.is_repetition()
                board.pop()
                
                # Apply repetition penalty/bonus based on whether this move leads to repetition
                if repetition_after_move:
                    repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                    # Check current position evaluation to determine if we're winning/losing
                    current_eval = self.evaluate(board)
                    # If we're winning and this move leads to repetition, heavily penalize it
                    # If we're losing and this move leads to repetition, prefer it
                    if current_eval < -20:  # Black is winning significantly
                        eval = 1000  # Heavy penalty for throwing away winning position
                    elif current_eval > 20:  # Black is losing significantly
                        eval = repetition_eval  # Prefer draw over losing
                
                # Update best move if this is better
                if eval < min_eval:
                    min_eval = eval
                    best_line = [move] + line
                    best_move = move
                
                # Alpha-beta pruning
                beta = min(beta, eval)
                if beta <= alpha:
                    # Store killer move before breaking
                    self._store_killer_move(move, absolute_depth, board)
                    break  # Alpha cutoff
            
                    
            # Exit node for visualization
            if self.viz_enabled:
                best_line_san = []
                if best_line:
                    temp_board = board.copy()
                    for move in best_line:
                        try:
                            best_line_san.append(temp_board.san(move))
                            temp_board.push(move)
                        except Exception:
                            best_line_san.append(move.uci())
                self.visualizer.exit_node(min_eval, "EXACT", best_line_san, 0, 0)
            else:
                self.visualizer.exit_node(min_eval, "EXACT", [], 0, 0)
            return min_eval, best_line, SearchStatus.COMPLETE

    def evaluate(self, board, game_stage=None):
        """
        Evaluate the current board position using the evaluation manager.
        
        Args:
            board: Current board state
            game_stage: Game stage (OPENING, MIDDLEGAME, or ENDGAME). If None, will be determined automatically.
            
        Returns:
            Evaluation score from White's perspective (rounded to 2 decimal places)
        """
        evaluation = self.evaluation_manager.evaluate(board, game_stage)
        return round(evaluation, 2)
    
    def evaluate_with_components(self, board, game_stage=None):
        """
        Evaluate the current board position with component breakdown.
        
        Args:
            board: Current board state
            game_stage: Game stage (OPENING, MIDDLEGAME, or ENDGAME). If None, will be determined automatically.
            
        Returns:
            Dictionary with evaluation components and total score
        """
        return self.evaluation_manager.evaluate_with_components(board, game_stage)
    
    
    def get_evaluator_info(self):
        """Get information about the current evaluator"""
        return self.evaluation_manager.get_evaluator_info()
    
    def switch_evaluator(self, evaluator_type: str, **kwargs):
        """Switch to a different evaluator"""
        self.evaluation_manager.switch_evaluator(evaluator_type, **kwargs)
    
    

    def _quiescence(self, board, alpha, beta, depth=0, game_stage=None):
        """
        Quiescence search - continues searching captures and checks to avoid horizon effect.
        
        Key improvements:
        - If position is in check, search ALL legal moves (not just captures)
        - If position is not in check, search captures and checks
        - This ensures tactical sequences are properly evaluated
        
        Args:
            board: Current board state
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Current quiescence depth (to prevent infinite loops)
            
        Returns:
            Tuple of (evaluation, principal_variation, status)
            - evaluation: The evaluation value (None if status is PARTIAL)
            - principal_variation: The best line (empty list if status is PARTIAL)
            - status: SearchStatus.COMPLETE or SearchStatus.PARTIAL
        """
        # Count this node
        self.nodes_searched += 1
        
        # Check for game over conditions first
        # NOTE: THIS CHECK IS COMMENTED OUT AS IT FEELS UNNECESSARY
        if False and board.is_game_over():
            if board.is_checkmate():
                # Checkmate evaluation
                checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                if board.turn:  # Black wins (White to move but checkmated)
                    return -checkmate_bonus, [], SearchStatus.COMPLETE
                else:  # White wins (Black to move but checkmated)
                    return checkmate_bonus, [], SearchStatus.COMPLETE
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
                # Draw conditions (except repetition)
                draw_value = self.evaluation_manager.evaluator.config.get("draw_value", 0)
                return draw_value, [], SearchStatus.COMPLETE
            elif board.is_repetition():
                # 3-fold repetition - can be evaluated differently
                repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                return repetition_eval, [], SearchStatus.COMPLETE
            else:
                # Other game over conditions
                return self.evaluate(board, game_stage), [], SearchStatus.COMPLETE
            
        # Limit quiescence depth to prevent infinite loops
        # depth represents quiescence depth (starting at 0), limit is additional depth beyond base search
        # Derive max depth from the individual move type limits
        max_quiescence_depth = max(
            self.quiescence_additional_depth_limit_captures,
            self.quiescence_additional_depth_limit_checks,
            self.quiescence_additional_depth_limit_promotions,
            self.quiescence_additional_depth_limit_queen_defense,
            self.quiescence_additional_depth_limit_value_threshold
        )
        if depth > max_quiescence_depth:
            return self.evaluate(board, game_stage), [], SearchStatus.COMPLETE
        
        # Check if position is in tablebase for perfect evaluation
        if self.tablebase and self.is_tablebase_position(board):
            try:
                wdl = self.tablebase.get_wdl(board)
                if wdl is not None:
                    # Convert WDL to evaluation score
                    # WDL values are from the perspective of the side to move
                    # WDL: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
                    if board.turn:  # White to move
                        if wdl == 2:  # White wins
                            return 100000, [], SearchStatus.COMPLETE
                        elif wdl == 1:  # White cursed win
                            return 50000, [], SearchStatus.COMPLETE
                        elif wdl == 0:  # Draw
                            return 0, [], SearchStatus.COMPLETE
                        elif wdl == -1:  # White blessed loss
                            return -50000, [], SearchStatus.COMPLETE
                        elif wdl == -2:  # White loses
                            return -100000, [], SearchStatus.COMPLETE
                    else:  # Black to move
                        if wdl == 2:  # Black wins (bad for White)
                            return -100000, [], SearchStatus.COMPLETE
                        elif wdl == 1:  # Black cursed win (bad for White)
                            return -50000, [], SearchStatus.COMPLETE
                        elif wdl == 0:  # Draw
                            return 0, [], SearchStatus.COMPLETE
                        elif wdl == -1:  # Black blessed loss (good for White)
                            return 50000, [], SearchStatus.COMPLETE
                        elif wdl == -2:  # Black loses (good for White)
                            return 100000, [], SearchStatus.COMPLETE
            except Exception:
                # Fall back to standard evaluation if tablebase lookup fails
                pass
        
        
        # Check if position is in check
        is_in_check = board.is_check()
        
        if is_in_check:
            # IN CHECK: Search ALL legal moves (must respond to check)
            moves_to_search = list(board.legal_moves)
            
            if not moves_to_search:
                # If in check and no legal moves, it's checkmate
                return -100000 if board.turn else 100000, [], SearchStatus.COMPLETE
        else:
            # NOT IN CHECK: Evaluate stand pat and search captures/checks
            stand_pat = self.evaluate(board, game_stage)
            
            # Alpha-beta pruning at quiescence level
            if board.turn:  # White to move: maximize
                if stand_pat >= beta:
                    return beta, [], SearchStatus.COMPLETE
                if stand_pat + 990 < alpha:  # DELTA PRUNING TMP
                    return alpha, [], SearchStatus.COMPLETE

                alpha = max(alpha, stand_pat)
            else:  # Black to move: minimize
                if stand_pat <= alpha:
                    return alpha, [], SearchStatus.COMPLETE
                if stand_pat - 990 > beta:  # DELTA PRUNING TMP
                    return beta, [], SearchStatus.COMPLETE

                beta = min(beta, stand_pat)
            
            # Search captures, checks, and promotions for tactical accuracy
            # Determine which move types to include based on quiescence depth
            # The limits specify additional depth beyond base search, so check if quiescence depth is within limit
            include_captures = depth <= self.quiescence_additional_depth_limit_captures
            include_checks = depth <= self.quiescence_additional_depth_limit_checks
            include_promotions = depth <= self.quiescence_additional_depth_limit_promotions
            include_queen_defense = depth <= self.quiescence_additional_depth_limit_queen_defense
            include_value_threshold = depth <= self.quiescence_additional_depth_limit_value_threshold
            
            moves_to_search = self._get_quiescence_moves(
                board,
                include_captures=include_captures,
                include_checks=include_checks,
                include_promotions=include_promotions,
                include_queen_defense=include_queen_defense,
                include_value_threshold=include_value_threshold
            )
            
            if not moves_to_search:
                # If not in check and no captures/checks, return stand pat
                return stand_pat, [], SearchStatus.COMPLETE
        
        best_line = []
        best_move = None
        original_alpha = alpha
        original_beta = beta
        
        for move in moves_to_search:
            # Record move being considered for visualization
            self.visualizer.record_move_considered(move, board)
            
            # Make the move
            board.push(move)
            # Recursively search the resulting position
            score, line, status = self._quiescence(board, alpha, beta, depth + 1, game_stage)
            
            # If search was partial, propagate up immediately
            if status == SearchStatus.PARTIAL:
                board.pop()
                return None, [], SearchStatus.PARTIAL
            # Undo the move
            board.pop()
            
            if board.turn:  # White to move: maximize
                if score > alpha:
                    alpha = score
                    best_line = [move] + line
                    best_move = move
                if alpha >= beta:
                    break  # Beta cutoff
            else:  # Black to move: minimize
                if score < beta:
                    beta = score
                    best_line = [move] + line
                    best_move = move
                if beta <= alpha:
                    break  # Alpha cutoff
        
        
        if board.turn:
            return alpha, best_line, SearchStatus.COMPLETE
        else:
            return beta, best_line, SearchStatus.COMPLETE
    
    def _has_pawns_on_promotion_rank(self, board):
        """
        Check if there are pawns on the pre-promotion rank (7th rank for white, 2nd rank for black).
        This is a fast bitboard check to determine if promotions are possible.
        
        Args:
            board: Current board state
            
        Returns:
            True if there are pawns on the pre-promotion rank, False otherwise
        """
        turn = board.turn
        
        # Get pawns for the side to move
        pawns_bb = board.pieces(chess.PAWN, turn)
        
        if not pawns_bb:
            return False
        
        # Define promotion rank based on color
        if turn == chess.WHITE:
            promotion_rank = 6  # 7th rank (0-indexed)
        else:
            promotion_rank = 1  # 2nd rank (0-indexed)
        
        # Check if any pawns are on the promotion rank using bitboard mask
        promotion_rank_mask = chess.BB_RANKS[promotion_rank]
        return bool(pawns_bb & promotion_rank_mask)
    
    def _get_promotion_value(self, board, move):
        """
        Calculate the net value of a promotion move for sorting.
        
        Net value = value of promoted piece - value of pawn + value of captured piece (if any)
        Higher values are searched first to improve pruning.
        
        Args:
            board: Current board state
            move: The promotion move to evaluate
            
        Returns:
            Promotion net value for sorting
        """
        # Value of promoted piece
        promoted_piece_value = self.mvv_lva_values.get(move.promotion, 0)
        
        # Value of pawn being promoted (always 1)
        pawn_value = self.mvv_lva_values[chess.PAWN]
        
        # Value of captured piece (if any)
        captured_piece_value = 0
        if board.is_capture(move):
            captured_piece_type = board.piece_type_at(move.to_square)
            if captured_piece_type is not None:
                captured_piece_value = self.mvv_lva_values[captured_piece_type]
        
        # Net value = promoted piece - pawn + captured piece
        return promoted_piece_value - pawn_value + captured_piece_value
    
    def _get_capture_value(self, board, move):
        """
        Calculate the value of a capture move for MVV-LVA sorting.
        
        MVV-LVA = Most Valuable Victim - Least Valuable Attacker
        Higher values are searched first to improve pruning.
        
        Args:
            board: Current board state
            move: The capture move to evaluate
            
        Returns:
            Capture value for sorting
        """
        victim_piece_type = board.piece_type_at(move.to_square)
        attacker_piece_type = board.piece_type_at(move.from_square)
        
        if victim_piece_type is None or attacker_piece_type is None:
            return 0
        
        # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
        # Higher values = more promising captures
        return self.mvv_lva_values[victim_piece_type] * 10 - self.mvv_lva_values[attacker_piece_type]
    
    def _is_mate_found(self, best_value, best_line):
        """
        Check if a mate (checkmate) or mate in 2 (distance to mate = 1) was found.
        
        Args:
            best_value: Best evaluation value from search
            best_line: Principal variation (line of moves)
            
        Returns:
            Tuple of (is_mate, distance_to_mate) where:
            - is_mate: True if mate or mate in 2 was found
            - distance_to_mate: Distance to mate (0 = checkmate, 1 = mate in 2)
        """
        checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
        
        # Check if evaluation indicates a mate
        # Mate evaluations are: checkmate_bonus - distance_to_mate
        # So checkmate_bonus - 1 <= abs(eval) <= checkmate_bonus + 1000 (with some tolerance)
        abs_eval = abs(best_value)
        
        if abs_eval >= checkmate_bonus - 1 and abs_eval <= checkmate_bonus + 1000:
            # This is a mate evaluation
            # Distance to mate is encoded in the evaluation
            # Format: checkmate_bonus - distance_to_mate (for winning side)
            if best_value > 0:  # White is winning
                distance_to_mate = checkmate_bonus - best_value
            else:  # Black is winning (negative eval: -(checkmate_bonus - distance))
                distance_to_mate = checkmate_bonus + best_value  # best_value is negative
            
            # Ensure distance_to_mate is an integer and non-negative
            distance_to_mate = int(round(distance_to_mate))
            if distance_to_mate < 0:
                distance_to_mate = 0
            
            # Also check the line length as a backup
            if best_line:
                line_distance = len(best_line)
                # Use the minimum of the two (evaluation-based and line-based)
                distance_to_mate = min(distance_to_mate, line_distance)
            
            # Return True if mate (distance 0) or mate in 2 (distance 1)
            if distance_to_mate <= 1:
                return True, distance_to_mate
        
        return False, None
    
    def _is_clear_best_capture(self, board, best_move, shallow_search_top_move):
        """
        Check if a move is a clear best capture that warrants early termination.
        
        Criteria:
        1. Move was the top move from shallow search
        2. Move is the top move from first iterative deepening iteration
        3. Move is a capture
        4. Captured piece value - capturing piece value > -50 (allows knight/bishop captures,
           but blocks risky captures like queen capturing pawn)
        
        This heuristic helps save time in obvious positions where a capture is clearly best,
        without needing to search deeper. The -50 threshold filters out risky captures that
        might have hidden flaws discovered at deeper depths.
        
        Args:
            board: Current board state
            best_move: Best move from first iterative deepening iteration
            shallow_search_top_move: Top move from shallow search (depth 2)
            
        Returns:
            True if move meets all criteria for clear best capture, False otherwise
        """
        # Check if feature is enabled
        if not self.evaluation_manager.evaluator.config.get("clear_best_capture_enabled", True):
            return False
        
        # 1. Check if move was top from shallow search
        if best_move != shallow_search_top_move:
            return False
        
        # 2. Check if move is a capture
        if not board.is_capture(best_move):
            return False
        
        # 3. Check capture value threshold (captured - capturing > -50)
        # Get piece values from config
        piece_values = self.evaluation_manager.evaluator.config.get("piece_values", {})
        
        # Map piece types to config keys
        piece_type_to_key = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen"
        }
        
        # Get captured piece value
        captured_piece_type = board.piece_type_at(best_move.to_square)
        if captured_piece_type is None or captured_piece_type not in piece_type_to_key:
            return False
        
        captured_piece_key = piece_type_to_key[captured_piece_type]
        captured_piece_value = piece_values.get(captured_piece_key, 0)
        
        # Get capturing piece value
        capturing_piece_type = board.piece_type_at(best_move.from_square)
        if capturing_piece_type is None or capturing_piece_type not in piece_type_to_key:
            return False
        
        capturing_piece_key = piece_type_to_key[capturing_piece_type]
        capturing_piece_value = piece_values.get(capturing_piece_key, 0)
        
        # Check threshold: captured - capturing > -50
        capture_value_diff = captured_piece_value - capturing_piece_value
        threshold = self.evaluation_manager.evaluator.config.get("clear_best_capture_threshold", -50)
        
        if capture_value_diff <= threshold:
            return False
        
        # All criteria met
        return True
    

    
    def _get_sorted_moves_optimized(self, board, depth=0):
        """
        Optimized move generation and sorting with killer move support.
        
        Move order:
        1. Promotions (highest priority)
        2. Winning captures (MVV-LVA > 0)
        3. Killer moves
        4. Other captures (MVV-LVA <= 0)
        5. Checks
        6. Non-captures
        
        Args:
            board: Current board state
            depth: Current search depth (for killer moves)
            
        Returns:
            List of moves sorted by priority
        """
        legal_moves = board.legal_moves
        promotions = []
        winning_captures = []
        other_captures = []
        killer_moves = []
        checks = []
        non_captures = []
        
        # Get killer moves for this depth
        current_killers = self.killer_moves.get(depth, {})
        
        # Single pass to categorize moves
        for move in legal_moves:
            # Check for promotions first (highest priority)
            if move.promotion:
                # Store promotion with its net value for sorting
                promotion_value = self._get_promotion_value(board, move)
                promotions.append((move, promotion_value))
            elif board.is_capture(move):
                # Calculate MVV-LVA value for capture
                capture_value = self._get_capture_value(board, move)
                if capture_value > 0:
                    # Winning capture - high priority
                    winning_captures.append((move, capture_value))
                elif move in current_killers:
                    # Non-winning capture that is also a killer move - prioritize as killer
                    killer_moves.append((move, current_killers[move]))
                else:
                    # Non-winning capture that is not a killer move
                    other_captures.append((move, capture_value))
            elif move in current_killers:
                # Non-capture killer move
                killer_moves.append((move, current_killers[move]))
            elif self._is_check_fast(board, move):
                checks.append(move)
            else:
                non_captures.append(move)
        
        # Sort promotions by net value (highest first) - zero overhead if promotions list is empty
        if promotions:
            promotions.sort(key=lambda x: x[1], reverse=True)
        sorted_promotions = [move for move, _ in promotions]
        
        # Sort winning captures by MVV-LVA (highest first)
        winning_captures.sort(key=lambda x: x[1], reverse=True)
        sorted_winning_captures = [move for move, _ in winning_captures]
        
        # Sort other captures by MVV-LVA (highest first)
        other_captures.sort(key=lambda x: x[1], reverse=True)
        sorted_other_captures = [move for move, _ in other_captures]
        
        # Sort killer moves by cutoff count (highest first)
        killer_moves.sort(key=lambda x: x[1], reverse=True)
        sorted_killer_moves = [move for move, _ in killer_moves]
        
        # Track killer moves used for statistics
        if killer_moves:
            self.killer_moves_used += len(killer_moves)
        
        # Combine moves in the specified order: promotions first, then winning captures, then killer moves, then rest
        result = []
        result.extend(sorted_promotions)
        result.extend(sorted_winning_captures)
        result.extend(sorted_killer_moves)
        result.extend(sorted_other_captures)
        result.extend(checks)
        result.extend(non_captures)
        
        return result

    def _store_killer_move(self, move, depth, board):
        """Store a killer move that caused a beta cutoff at the given depth."""
        if depth <= 0:
            return  # Don't store killer moves at leaf nodes
        
        # Don't store captures or promotions as killer moves
        if move.promotion or board.is_capture(move):
            return
        
        # Initialize killer moves for this depth if not exists
        if depth not in self.killer_moves:
            self.killer_moves[depth] = {}
        
        killer_dict = self.killer_moves[depth]
        
        # Increment cutoff count for this move
        if move in killer_dict:
            killer_dict[move] += 1
        else:
            killer_dict[move] = 1
            self.killer_moves_stored += 1

    def _clear_killer_moves(self):
        """Clear all killer moves to avoid stale data from previous searches."""
        self.killer_moves.clear()
        self.killer_moves_stored = 0
        self.killer_moves_used = 0

    @staticmethod
    def predict_time(nodes: int, moves: int, num_pieces: int | None = None, game_stage: int | None = None) -> dict[int, float]:
        """
        Return predicted full-search time (seconds) at depths 4, 6, 8, 10.
        Per-depth linear models in (nodes, moves, piece-bucket) with 4x penalty
        on underestimates, plus multiplicative depth ratios:
          t6 >= r46 * t4, t8 >= r68 * t6, t10 >= r8_10 * t8.
        
        Args:
            nodes: Number of nodes searched in shallow search
            moves: Number of moves evaluated in shallow search
            num_pieces: Number of pieces on the board
            game_stage: Game stage (0=OPENING, 1=MIDDLEGAME, 2=ENDGAME) for logging
        """

        p = 32 if num_pieces is None else int(num_pieces)
        if p <= 16:
            bucket = "end"
        elif p <= 26:
            bucket = "mid"
        else:
            bucket = "open"

        def lin(nodes: int, moves: int, bucket: str,
                a: float, b: float, c_open: float, c_mid: float, c_end: float) -> float:
            c = c_open if bucket == "open" else (c_mid if bucket == "mid" else c_end)
            t = a * nodes + b * moves + c
            return max(0.05, float(t))

        # Depth-4 linear + scale
        a4, b4 = 0.0005335, -0.00820
        c4_open, c4_mid, c4_end = 0.221, -0.163, -0.196
        s4 = 1.44
        t4 = lin(nodes, moves, bucket, a4, b4, c4_open, c4_mid, c4_end) * s4

        # Depth-6 independent linear + scale
        a6, b6 = 0.000734, 0.1443
        c6_open, c6_mid, c6_end = 9.996, 5.012, 1.842
        s6 = 1.315
        t6_ind = lin(nodes, moves, bucket, a6, b6, c6_open, c6_mid, c6_end) * s6

        # Depth-8 independent linear + scale
        a8, b8 = 0.00388, 0.1977
        c8_open, c8_mid, c8_end = 3.864, 5.912, 5.679
        s8 = 1.345
        t8_ind = lin(nodes, moves, bucket, a8, b8, c8_open, c8_mid, c8_end) * s8

        # Depth-10 independent linear + scale
        a10, b10 = -0.00443, -2.354
        c10_open, c10_mid, c10_end = 8.338, 8.338, 16.675
        s10 = 1.0
        t10_ind = lin(nodes, moves, bucket, a10, b10, c10_open, c10_mid, c10_end) * s10

        # Multiplicative depth ratios (trained)
        r46 = 2.78
        r68 = 1.0
        r8_10 = 1.0

        t6_mult = r46 * t4
        t6 = max(t6_ind, t6_mult)

        t8_mult = r68 * t6
        t8 = max(t8_ind, t8_mult)

        t10_mult = r8_10 * t8
        t10 = max(t10_ind, t10_mult)

        return {4: float(t4), 6: float(t6), 8: float(t8), 10: float(t10)}

    def _predict_optimal_starting_depth(self, shallow_search_stats, time_budget, starting_depth, game_stage=None):
        """
        Predict the optimal starting depth for iterative deepening based on shallow search results.
        
        Args:
            shallow_search_stats: Dictionary with 'nodes', 'moves', and 'num_pieces' from shallow search
            time_budget: Available time budget in seconds
            starting_depth: Min starting depth from configuration
            game_stage: Game stage (0=OPENING, 1=MIDDLEGAME, 2=ENDGAME) for logging
        Returns:
            Optimal starting depth
        """
        if not shallow_search_stats or 'nodes' not in shallow_search_stats or 'moves' not in shallow_search_stats or 'num_pieces' not in shallow_search_stats:
            # Fallback to default if no shallow search stats available
            return self.evaluation_manager.evaluator.config.get("search_depth_starting", 3)
        
        nodes = shallow_search_stats['nodes']
        moves = shallow_search_stats['moves']
        num_pieces = shallow_search_stats['num_pieces']
        
        # Get predicted times for different depths
        predicted_times = self.predict_time(nodes, moves, num_pieces, game_stage)
        
        # Find the highest depth we can complete within the time budget
        # Use a safety factor to ensure we don't exceed the budget
        safety_factor = self.evaluation_manager.evaluator.config.get("predictive_time_safety_factor", 0.8)
        available_time = time_budget * safety_factor
        
        optimal_depth = starting_depth  # Default fallback
        
        # TODO: DISABLED
        #for depth in [starting_depth]:
        #    if predicted_times[depth] <= available_time:
        #        optimal_depth = depth
        #        break
        
        # Convert game_stage to string for logging
        game_stage_str = "OPENING" if game_stage == 0 else ("MIDDLEGAME" if game_stage == 1 else ("ENDGAME" if game_stage == 2 else "UNKNOWN"))
        
        if not self.quiet:
            self.logger.log_info(f"Predictive time management: nodes={nodes}, moves={moves}, pieces={num_pieces}, game_stage={game_stage_str}")
            self.logger.log_info(f"Predicted times: depth 4={predicted_times[4]:.2f}s, depth 6={predicted_times[6]:.2f}s, depth 8={predicted_times[8]:.2f}s, depth 10={predicted_times[10]:.2f}s")
            self.logger.log_info(f"Time budget: {time_budget:.2f}s, available: {available_time:.2f}s, optimal depth: {optimal_depth}")
        
        return optimal_depth

    def get_killer_move_stats(self):
        """Get killer move statistics."""
        total_cutoffs = sum(sum(counts.values()) for counts in self.killer_moves.values())
        return {
            'killer_moves_stored': self.killer_moves_stored,
            'killer_moves_used': self.killer_moves_used,
            'killer_depths': len(self.killer_moves),
            'total_killer_entries': sum(len(moves) for moves in self.killer_moves.values()),
            'total_cutoffs': total_cutoffs
        }

    def test_capture_evaluation(self, board, move):
        """
        Test function to verify capture evaluation.
        
        Args:
            board: Current board state
            move: The capture move to test
            
        Returns:
            Evaluation after the capture
        """
        print(f"\n🧪 Testing capture: {board.san(move)}")
        print(f"📋 Board before move:")
        print(board)
        
        # Count pieces before
        white_pieces_before = {chess.PAWN: len(board.pieces(chess.PAWN, chess.WHITE)),
                              chess.KNIGHT: len(board.pieces(chess.KNIGHT, chess.WHITE)),
                              chess.BISHOP: len(board.pieces(chess.BISHOP, chess.WHITE)),
                              chess.ROOK: len(board.pieces(chess.ROOK, chess.WHITE)),
                              chess.QUEEN: len(board.pieces(chess.QUEEN, chess.WHITE))}
        black_pieces_before = {chess.PAWN: len(board.pieces(chess.PAWN, chess.BLACK)),
                              chess.KNIGHT: len(board.pieces(chess.KNIGHT, chess.BLACK)),
                              chess.BISHOP: len(board.pieces(chess.BISHOP, chess.BLACK)),
                              chess.ROOK: len(board.pieces(chess.ROOK, chess.BLACK)),
                              chess.QUEEN: len(board.pieces(chess.QUEEN, chess.BLACK))}
        
        print(f"⚪ White pieces before: {white_pieces_before}")
        print(f"⚫ Black pieces before: {black_pieces_before}")
        
        # Make the move
        board.push(move)
        
        print(f"📋 Board after move:")
        print(board)
        
        # Count pieces after
        white_pieces_after = {chess.PAWN: len(board.pieces(chess.PAWN, chess.WHITE)),
                             chess.KNIGHT: len(board.pieces(chess.KNIGHT, chess.WHITE)),
                             chess.BISHOP: len(board.pieces(chess.BISHOP, chess.WHITE)),
                             chess.ROOK: len(board.pieces(chess.ROOK, chess.WHITE)),
                             chess.QUEEN: len(board.pieces(chess.QUEEN, chess.WHITE))}
        black_pieces_after = {chess.PAWN: len(board.pieces(chess.PAWN, chess.BLACK)),
                             chess.KNIGHT: len(board.pieces(chess.KNIGHT, chess.BLACK)),
                             chess.BISHOP: len(board.pieces(chess.BISHOP, chess.BLACK)),
                             chess.ROOK: len(board.pieces(chess.ROOK, chess.BLACK)),
                             chess.QUEEN: len(board.pieces(chess.QUEEN, chess.BLACK))}
        
        print(f"⚪ White pieces after: {white_pieces_after}")
        print(f"⚫ Black pieces after: {black_pieces_after}")
        
        # Evaluate the position
        eval_after = self.evaluate(board)
        if not self.quiet:
            print(f"📊 Evaluation after move: {eval_after}")
        
        # Undo the move
        board.pop()
        
        return eval_after
    
    def is_tablebase_position(self, board):
        """Check if position is suitable for tablebase lookup"""
        return chess.popcount(board.occupied) <= 5
    
    
    def get_tablebase_move(self, board):
        """Get the best move from tablebase if available"""
        try:
            if not self.tablebase:
                if not self.quiet:
                    self.logger.log("⚠️  No tablebase available")
                return None
            
            # Get tablebase information for current position
            wdl = self.tablebase.get_wdl(board)
            if wdl is None:
                if not self.quiet:
                    self.logger.log("⚠️  Position not found in tablebase")
                return None
            
            # WDL values: 2 = win, 1 = cursed win, 0 = draw, -1 = blessed loss, -2 = loss
            # NOTE: The WDL value is from the perspective of side that will move next (after this move)
            wdl_names = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
            if not self.quiet:
                self.logger.log(f"📊 Tablebase: WDL={wdl} ({wdl_names.get(wdl, 'Unknown')})")
            
            # Get DTZ for current position to understand the fastest path
            dtz = self.tablebase.get_dtz(board)
            if dtz is not None and not self.quiet:
                self.logger.log(f"📊 Tablebase: DTZ={dtz} (Distance To Zero)")
            
            # Find the best move by trying each legal move
            # We'll collect all moves with their WDL/DTZ values and then apply our selection logic
            move_candidates = []
            
            if not self.quiet:
                self.logger.log(f"🔍 Checking {len(list(board.legal_moves))} legal moves...")
            
            moves_checked = 0
            for move in board.legal_moves:
                # Get SAN notation before pushing the move
                move_san = board.san(move)
                
                # Check if this is a pawn move or capture before making the move
                is_pawn_move = board.piece_type_at(move.from_square) == chess.PAWN
                is_capture = board.is_capture(move)
                
                board.push(move)
                try:
                    move_wdl = self.tablebase.get_wdl(board)
                    move_dtz = self.tablebase.get_dtz(board)
                    moves_checked += 1
                    
                    if move_wdl is not None:
                        
                        move_candidates.append({
                            'move': move,
                            'san': move_san,
                            'wdl': move_wdl,
                            'dtz': move_dtz,
                            'is_pawn_move': is_pawn_move,
                            'is_capture': is_capture
                        })
                        
                        if not self.quiet:
                            self.logger.log(f"  ✅ {move_san}: WDL={move_wdl} ({wdl_names.get(move_wdl, 'Unknown')}), DTZ={move_dtz}")
                    else:
                        if not self.quiet:
                            self.logger.log(f"  ❌ {move_san}: Not found in tablebase")
                except Exception as e:
                    if not self.quiet:
                        self.logger.log(f"⚠️  Error checking move {move_san}: {e}")
                board.pop()
            
            # Now apply our selection logic
            if move_candidates:
                # Sort by WDL first (lower is better for the opponent)
                # Then by pawn moves and captures
                # Then by DTZ (higher is better)
                def move_priority(candidate):
                    wdl = candidate['wdl']
                    is_pawn_or_capture = candidate['is_pawn_move'] or candidate['is_capture']
                    dtz = candidate['dtz'] if candidate['dtz'] is not None else -999
                    
                    return (wdl, -is_pawn_or_capture, -dtz)
                
                move_candidates.sort(key=move_priority)
                best_candidate = move_candidates[0]
                
                best_move = best_candidate['move']
                best_wdl = best_candidate['wdl']
                best_dtz = best_candidate['dtz']
                
                if not self.quiet:
                    side_to_move = "White" if board.turn else "Black"
                    self.logger.log(f"🎯 Selected {best_candidate['san']} for {side_to_move} (WDL={best_wdl}, DTZ={best_dtz}, Pawn/Capture={best_candidate['is_pawn_move'] or best_candidate['is_capture']})")
            else:
                best_move = None
                best_wdl = None
                best_dtz = None
            
            if not self.quiet:
                self.logger.log(f"📊 Checked {moves_checked} moves in tablebase")
            
            if best_move:
                best_move_san = board.san(best_move)
                side_to_move = "White" if board.turn else "Black"
                if not self.quiet:
                    self.logger.log(f"🎯 Tablebase best move for {side_to_move}: {best_move_san} (WDL={best_wdl} - {wdl_names.get(best_wdl, 'Unknown')}, DTZ={best_dtz})")
            else:
                if not self.quiet:
                    self.logger.log("⚠️  No best move found in tablebase")
            
            return best_move
            
        except Exception as e:
            if not self.quiet:
                self.logger.log(f"⚠️  Tablebase error: {e}")
            return None 

    def _get_quiescence_moves(self, board, include_captures=True, include_checks=False, 
                               include_promotions=False, include_queen_defense=False, include_value_threshold=False):
        """
        Get moves to search in quiescence, ordered by priority:
        1. Promotions (highest priority)
        2. Captures (ordered by MVV-LVA)
        3. Checks
        4. Defensive moves (if configured)
        
        Args:
            board: Current board state
            include_captures: Whether to include captures
            include_checks: Whether to include checks
            include_promotions: Whether to include promotion moves
            include_queen_defense: Whether to include queen defensive moves
            include_value_threshold: Whether to include value threshold defensive moves
            
        Returns:
            List of moves to search in quiescence, ordered by priority
        """
        # Check if promotions are possible (fast bitboard check)
        has_promotions = include_promotions and self._has_pawns_on_promotion_rank(board)
        
        # Collect promotions first if enabled and available (highest priority)
        promotions = []
        legal_moves = None  # Will be set if we need to generate all legal moves
        
        if has_promotions:
            # Generate all legal moves once (we'll need them for categorization anyway)
            legal_moves = list(board.legal_moves)
            promotions = [move for move in legal_moves if move.promotion]
        
        # Optimization: if only captures are needed (and no promotions), use the faster generate_legal_captures()
        # If promotions are possible, we need to generate all legal moves anyway, so this optimization doesn't apply
        if include_captures and not include_checks and not include_queen_defense and not include_value_threshold and not has_promotions:
            captures = list(board.generate_legal_captures())
            # Order captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            captures.sort(key=lambda move: self._get_capture_value(board, move), reverse=True)
            return promotions + captures
        
        # Otherwise, generate all legal moves and categorize (if not already generated)
        if legal_moves is None:
            legal_moves = list(board.legal_moves)
        
        # Categorize moves by type
        captures = []
        checks = []
        defensive_moves = []
        
        for move in legal_moves:
            # Skip promotions if we already collected them separately
            if has_promotions and move.promotion:
                continue
            
            is_capture = board.is_capture(move)
            
            if is_capture and include_captures:
                # Include captures
                captures.append(move)
            elif not is_capture:
                # Check if this non-capture move gives check
                if include_checks and board.gives_check(move):
                    checks.append(move)
                # Check if this is a defensive move (queen defense)
                elif include_queen_defense:
                    if self._is_defensive_move_fast(board, move, include_queen_defense=True, include_value_threshold=False):
                        defensive_moves.append(move)
                # Check if this is a defensive move (value threshold)
                elif include_value_threshold:
                    if self._is_defensive_move_fast(board, move, include_queen_defense=False, include_value_threshold=True):
                        defensive_moves.append(move)
        
        # Order captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        captures.sort(key=lambda move: self._get_capture_value(board, move), reverse=True)
        
        # Combine moves in priority order: promotions first, then captures, then checks, then defensive moves
        moves_to_search = promotions + captures + checks + defensive_moves
        
        return moves_to_search
    
    def _is_check_fast(self, board, move):
        """
        Quickly determine if a move gives check without making the move.
        This primarily checks for direct checks on the move's destination square.
        
        Note: This method might not detect all *discovered* checks perfectly
        without a more complex bitboard implementation, but it is much faster
        than board.gives_check() and captures most cases.
        
        Args:
            board: Current board state
            move: The move to check
            
        Returns:
            True if the move gives check (approximately), False otherwise
        """
        # Identify the opponent's king square
        opponent_color = not board.turn
        opponent_king_square = board.king(opponent_color)
        if opponent_king_square is None:
            # Should not happen in a normal game, but handle for safety
            return False
        
        # Get the piece type of the moving piece
        piece_type = board.piece_type_at(move.from_square)
        if piece_type is None:
            return False
        
        # Check if the destination square of the move directly attacks the opponent's king square
        # This uses the built-in attack generators of the python-chess library
        if piece_type == chess.PAWN:
            return bool(chess.BB_PAWN_ATTACKS[board.turn][move.to_square] & chess.BB_SQUARES[opponent_king_square])
        elif piece_type == chess.KNIGHT:
            return bool(chess.BB_KNIGHT_ATTACKS[move.to_square] & chess.BB_SQUARES[opponent_king_square])
        elif piece_type == chess.BISHOP:
            # For sliding pieces, check if king is on same diagonal and path is clear
            ray_bb = chess.ray(move.to_square, opponent_king_square)
            if ray_bb and chess.square_distance(move.to_square, opponent_king_square) > 0:
                # Check if the path is clear (no pieces blocking)
                between_bb = chess.between(move.to_square, opponent_king_square)
                return (between_bb & board.occupied) == 0
            return False
        elif piece_type == chess.ROOK:
            # Check if king is on same rank or file and path is clear
            ray_bb = chess.ray(move.to_square, opponent_king_square)
            if ray_bb and chess.square_distance(move.to_square, opponent_king_square) > 0:
                between_bb = chess.between(move.to_square, opponent_king_square)
                return (between_bb & board.occupied) == 0
            return False
        elif piece_type == chess.QUEEN:
            # Queen can move like bishop or rook
            ray_bb = chess.ray(move.to_square, opponent_king_square)
            if ray_bb and chess.square_distance(move.to_square, opponent_king_square) > 0:
                between_bb = chess.between(move.to_square, opponent_king_square)
                return (between_bb & board.occupied) == 0
            return False
        elif piece_type == chess.KING:
            # A king move cannot give check to the other king unless the other king is on an adjacent square,
            # which is an illegal position anyway. So this can safely return False.
            return False
        
        return False
    
    def _is_defensive_move_fast(self, board, move, include_queen_defense=False, include_value_threshold=False):
        """
        Fast defensive move check using provided configuration flags.
        
        Args:
            board: Current board state
            move: The move to check
            include_queen_defense: Whether to check for queen defensive moves
            include_value_threshold: Whether to check for value threshold defensive moves
            
        Returns:
            True if the move is defensive, False otherwise
        """
        if move.from_square is None:
            return False
        
        piece = board.piece_at(move.from_square)
        if piece is None or piece.color != board.turn:
            return False
        
        # Check if the piece is under attack
        if not board.is_attacked_by(not board.turn, move.from_square):
            return False
        
        # Queen defense: only consider queen defensive moves
        if include_queen_defense and piece.piece_type == chess.QUEEN:
            return True
        
        # Value threshold: consider moves for pieces above value threshold
        if include_value_threshold:
            piece_values = self.evaluation_manager.evaluator.config.get("piece_values", {})
            piece_symbol = piece.symbol().upper()
            if piece_symbol in piece_values:
                piece_value = piece_values[piece_symbol]
                if piece_value >= self.quiescence_value_threshold:
                    return True
        
        return False
