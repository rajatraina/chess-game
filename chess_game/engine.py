import random
import chess
import chess.syzygy
import time
import os
import zlib
try:
    from .evaluation import EvaluationManager, create_evaluator
    from .logging_manager import get_logger
    from .search_visualizer import get_visualizer, configure_visualizer
except ImportError:
    from evaluation import EvaluationManager, create_evaluator
    from logging_manager import get_logger
    from search_visualizer import get_visualizer, configure_visualizer

# Transposition table entry types
EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

class TranspositionTableEntry:
    """Represents a single entry in the transposition table"""
    def __init__(self, hash_key, depth, score, best_move, node_type, age):
        self.hash_key = hash_key
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type  # EXACT, LOWER_BOUND, UPPER_BOUND
        self.age = age
    
    def __str__(self):
        node_type_str = {EXACT: "EXACT", LOWER_BOUND: "LOWER", UPPER_BOUND: "UPPER"}[self.node_type]
        return f"TT[{self.hash_key:016x}] d={self.depth} s={self.score} m={self.best_move} t={node_type_str} a={self.age}"

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
    
    def __init__(self, depth=None, evaluator_type="handcrafted", evaluator_config=None, quiet=False, new_best_move_callback=None):
        self.nodes_searched = 0
        self.search_start_time = 0
        self.quiet = quiet  # Control debug output
        self.new_best_move_callback = new_best_move_callback
        
        # Initialize logging manager
        self.logger = get_logger(new_best_move_callback, quiet)
        
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
        
        # Evaluation cache for repeated positions
        self.eval_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Transposition table
        self.transposition_table = {}
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_cutoffs = 0
        self.search_age = 0
        
        # TT configuration
        self.tt_size_limit = 1000000  # Maximum number of entries
        self.tt_enabled = True
        
        # Initialize Zobrist hash for transposition table
        self._init_zobrist_hash()
        
        # Load time management parameters
        self.time_management_moves_remaining = self.evaluation_manager.evaluator.config.get("time_management_moves_remaining", 30)
        self.time_management_increment_used = self.evaluation_manager.evaluator.config.get("time_management_increment_used", 0.8)
        
        # Load move ordering parameters
        self.moveorder_shallow_search_depth = self.evaluation_manager.evaluator.config.get("moveorder_shallow_search_depth", 2)
        self.moveorder_shallow_search_quiescence_depth_limit = self.evaluation_manager.evaluator.config.get("moveorder_shallow_search_quiescence_depth_limit", 2)
        
        # Initialize search visualizer
        self.visualizer = get_visualizer()
        viz_enabled = self.evaluation_manager.evaluator.config.get("search_visualization_enabled", False)
        viz_target_fen = self.evaluation_manager.evaluator.config.get("search_visualization_target_fen", None)
        configure_visualizer(viz_enabled, viz_target_fen)
        
        # Move counter for visualization
        self.move_counter = 0
    
    def calculate_time_budget(self, time_remaining, increment):
        """
        Calculate time budget for the current move.
        
        Args:
            time_remaining: Time remaining in milliseconds
            increment: Time increment in milliseconds
            
        Returns:
            Time budget in seconds
        """
        # Convert to seconds
        time_remaining_sec = time_remaining / 1000.0
        increment_sec = increment / 1000.0
        
        # Calculate time budget using the formula
        time_budget = (time_remaining_sec / self.time_management_moves_remaining) + (increment_sec * self.time_management_increment_used)
        
        # Ensure minimum time budget of 0.1 seconds
        return max(time_budget, 0.1)
    
    def _shallow_search_with_quiescence(self, board, depth, alpha, beta):
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
                    return -checkmate_bonus, []
                else:  # White wins (Black to move but checkmated)
                    return checkmate_bonus, []
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
                draw_value = self.evaluation_manager.evaluator.config.get("draw_value", 0)
                return draw_value, []  # Draw
            elif board.is_repetition():
                repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                return repetition_eval, []  # 3-fold repetition
        
        # Leaf node: use quiescence search
        if depth == 0:
            return self._quiescence_shallow(board, alpha, beta, 0)
        
        # Probe transposition table
        tt_score, tt_best_move, tt_node_type = self._probe_transposition(board, depth, alpha, beta)
        
        # If we have a usable transposition table entry, return it
        if tt_score is not None:
            return tt_score, [tt_best_move] if tt_best_move else []
        
        # Check if position is in tablebase for perfect evaluation
        if self.tablebase and self.is_tablebase_position(board):
            try:
                wdl = self.tablebase.get_wdl(board)
                if wdl is not None:
                    # Convert WDL to evaluation score
                    if wdl == 2:  # Win
                        return 100000, []
                    elif wdl == 1:  # Cursed win
                        return 50000, []
                    elif wdl == 0:  # Draw
                        return 0, []
                    elif wdl == -1:  # Blessed loss
                        return -50000, []
                    elif wdl == -2:  # Loss
                        return -100000, []
            except Exception:
                pass
        
        # Get sorted moves for this shallow search
        sorted_moves = self._get_sorted_moves_optimized(board, tt_best_move)
        
        best_value = -float('inf') if board.turn else float('inf')
        best_line = []
        
        for move in sorted_moves:
            # Make the move
            board.push(move)
            
            # Search the resulting position
            value, line = self._shallow_search_with_quiescence(board, depth - 1, alpha, beta)
            
            # Undo the move
            board.pop()
            
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
        
        # Store in transposition table
        node_type = EXACT
        if best_value <= alpha:
            node_type = UPPER_BOUND
        elif best_value >= beta:
            node_type = LOWER_BOUND
        
        self._store_transposition(board, depth, best_value, best_line[0] if best_line else None, node_type)
        
        return best_value, best_line
    
    def _quiescence_shallow(self, board, alpha, beta, depth=0):
        """
        Limited quiescence search for shallow move ordering.
        
        Args:
            board: Current board state
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Current quiescence depth
            
        Returns:
            Tuple of (evaluation, principal_variation)
        """
        # Count this node
        self.nodes_searched += 1
        
        # Check for game over conditions
        if board.is_game_over():
            return self.evaluate(board), []
        
        # Limit quiescence depth
        if depth > self.moveorder_shallow_search_quiescence_depth_limit:
            return self.evaluate(board), []
        
        # Check if position is in check
        is_in_check = board.is_check()
        
        # Evaluate current position (stand pat) - only if not in check
        if not is_in_check:
            stand_pat = self.evaluate_cached(board)
            
            # Alpha-beta pruning at quiescence level
            if board.turn:  # White to move: maximize
                if stand_pat >= beta:
                    return beta, []
                alpha = max(alpha, stand_pat)
            else:  # Black to move: minimize
                if stand_pat <= alpha:
                    return alpha, []
                beta = min(beta, stand_pat)
        
        # Determine which moves to search
        if is_in_check:
            # If in check, search ALL legal moves
            moves_to_search = list(board.legal_moves)
        else:
            # If not in check, search captures and checks
            moves_to_search = self._get_quiescence_moves(board)
        
        if not moves_to_search:
            if is_in_check:
                # If in check and no legal moves, it's checkmate
                return -100000 if board.turn else 100000, []
            else:
                # If not in check and no captures/checks, return stand pat
                stand_pat = self.evaluate_cached(board)
                return stand_pat, []
        
        best_line = []
        best_move = None
        
        for move in moves_to_search:
            # Make the move
            board.push(move)
            
            # Recursive quiescence search
            value, line = self._quiescence_shallow(board, alpha, beta, depth + 1)
            
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
        
        return (alpha if board.turn else beta), best_line
    
    def _order_moves_with_shallow_search(self, board):
        """
        Order moves using shallow search with limited quiescence.
        
        Args:
            board: Current board state
            
        Returns:
            List of moves ordered by shallow search evaluation
        """
        if not self.quiet:
            print(f"🔍 Performing shallow search (depth {self.moveorder_shallow_search_depth}) for move ordering...")
        
        # Store original node count
        original_nodes = self.nodes_searched
        
        # Perform shallow search on all legal moves
        move_evaluations = []
        legal_moves = list(board.legal_moves)
        
        for move in legal_moves:
            # Make the move
            board.push(move)
            
            # Perform shallow search
            eval_score, _ = self._shallow_search_with_quiescence(
                board, 
                self.moveorder_shallow_search_depth - 1, 
                -float('inf'), 
                float('inf')
            )
            
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
        
        # Log shallow search statistics and move order
        shallow_nodes = self.nodes_searched - original_nodes
        self.logger.log_shallow_search_stats(shallow_nodes, len(legal_moves))
        self.logger.log_move_order(board, ordered_moves)
        
        return ordered_moves
    
    def _clean_transposition_table_after_shallow_search(self):
        """
        Clean the transposition table after shallow search to remove low-depth entries.
        This prevents TT pollution from shallow search entries.
        """
        if not self.tt_enabled:
            return
        
        # Determine cutoff depth (remove entries from shallow search)
        cutoff_depth = self.depth - 1
        
        # Count entries before cleaning
        entries_before = len(self.transposition_table)
        
        # Remove shallow entries
        keys_to_remove = []
        for key, entry in self.transposition_table.items():
            if entry.depth < cutoff_depth:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.transposition_table[key]
        
        # Log cleaning statistics
        entries_after = len(self.transposition_table)
        self.logger.log_tt_cleaning(entries_before, entries_after)
    
    def _init_zobrist_hash(self):
        """Initialize Zobrist hash tables for efficient position hashing"""
        import random
        random.seed(42)  # For reproducible hashes
        
        # Piece-square hash table
        self.zobrist_pieces = {}
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                for square in chess.SQUARES:
                    self.zobrist_pieces[(piece_type, color, square)] = random.getrandbits(64)
        
        # Castling rights hash table - iterate through all possible castling combinations
        self.zobrist_castling = {}
        for i in range(16):  # 4 bits for castling rights (KQkq)
            self.zobrist_castling[i] = random.getrandbits(64)
        
        # En passant square hash table
        self.zobrist_en_passant = {}
        for square in chess.SQUARES:
            self.zobrist_en_passant[square] = random.getrandbits(64)
        
        # Side to move hash
        self.zobrist_side = random.getrandbits(64)
    
    def _get_zobrist_hash(self, board):
        """Generate Zobrist hash for the current board position"""
        hash_key = 0
        
        # Hash pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                hash_key ^= self.zobrist_pieces[(piece.piece_type, piece.color, square)]
        
        # Hash castling rights - convert to integer index
        castling_index = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_index |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_index |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_index |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_index |= 8
        hash_key ^= self.zobrist_castling[castling_index]
        
        # Hash en passant square
        if board.ep_square is not None:
            hash_key ^= self.zobrist_en_passant[board.ep_square]
        
        # Hash side to move
        if board.turn:
            hash_key ^= self.zobrist_side
        
        return hash_key
    
    def _store_transposition(self, board, depth, score, best_move, node_type):
        """Store a position in the transposition table"""
        if not self.tt_enabled:
            return
        
        hash_key = self._get_zobrist_hash(board)
        
        # Create new entry
        entry = TranspositionTableEntry(hash_key, depth, score, best_move, node_type, self.search_age)
        
        # Replace existing entry or add new one
        self.transposition_table[hash_key] = entry
        
        # Manage table size
        if len(self.transposition_table) > self.tt_size_limit:
            self._cleanup_transposition_table()
    
    def _probe_transposition(self, board, depth, alpha, beta):
        """Probe the transposition table for a position"""
        if not self.tt_enabled:
            return None, None, None
        
        hash_key = self._get_zobrist_hash(board)
        
        if hash_key not in self.transposition_table:
            self.tt_misses += 1
            return None, None, None
        
        entry = self.transposition_table[hash_key]
        self.tt_hits += 1
        
        # Check if the stored entry is deep enough
        if entry.depth >= depth:
            score = entry.score
            best_move = entry.best_move
            
            if entry.node_type == EXACT:
                return score, best_move, entry.node_type
            elif entry.node_type == LOWER_BOUND and score >= beta:
                self.tt_cutoffs += 1
                return score, best_move, entry.node_type
            elif entry.node_type == UPPER_BOUND and score <= alpha:
                self.tt_cutoffs += 1
                return score, best_move, entry.node_type
        
        return None, entry.best_move, None  # Return best move for move ordering even if score not usable
    
    def _cleanup_transposition_table(self):
        """Clean up transposition table by removing old entries"""
        if len(self.transposition_table) <= self.tt_size_limit:
            return
        
        # Remove entries older than current search age - 2
        cutoff_age = max(0, self.search_age - 2)
        keys_to_remove = []
        
        for hash_key, entry in self.transposition_table.items():
            if entry.age < cutoff_age:
                keys_to_remove.append(hash_key)
        
        # Remove old entries
        for key in keys_to_remove:
            del self.transposition_table[key]
        
        # If still too large, remove random entries
        if len(self.transposition_table) > self.tt_size_limit:
            keys = list(self.transposition_table.keys())
            keys_to_remove = random.sample(keys, len(keys) // 4)  # Remove 25%
            for key in keys_to_remove:
                del self.transposition_table[key]
    
    def get_transposition_stats(self):
        """Get transposition table statistics"""
        total_probes = self.tt_hits + self.tt_misses
        hit_rate = (self.tt_hits / total_probes * 100) if total_probes > 0 else 0
        cutoff_rate = (self.tt_cutoffs / self.tt_hits * 100) if self.tt_hits > 0 else 0
        
        return {
            'hits': self.tt_hits,
            'misses': self.tt_misses,
            'cutoffs': self.tt_cutoffs,
            'total_probes': total_probes,
            'hit_rate': hit_rate,
            'cutoff_rate': cutoff_rate,
            'table_size': len(self.transposition_table),
            'table_size_limit': self.tt_size_limit
        }
    
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
                print(f"⚠️  Could not initialize tablebase: {e}")
            self.tablebase = None

    def get_move(self, board, time_budget=None, repetition_detected=False):
        """
        Find the best move for the current side to move.
        
        Args:
            board: Current chess board state
            time_budget: Maximum time to spend on this move in seconds (None for unlimited)
            repetition_detected: Whether the current position is a 3-fold repetition
            
        Returns:
            Best move found by the search
        """
        # Increment search age for transposition table management
        self.search_age += 1
        
        # Reset transposition table statistics for this search
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_cutoffs = 0
        
        # Set the starting position for consistent evaluation throughout search
        if hasattr(self.evaluation_manager.evaluator, '_set_starting_position'):
            self.evaluation_manager.evaluator._set_starting_position(board)
        
        # Check if this is an endgame position for deeper search
        is_endgame = self._is_endgame_position(board)
        search_depth = self.depth
        
        if is_endgame:
            endgame_depth = self.evaluation_manager.evaluator.config.get("endgame_search_depth", 6)
            search_depth = max(self.depth, endgame_depth)
            if not self.quiet:
                print(f"🎯 Endgame detected - using depth {search_depth}")
        best_move = None
        # Initialize best_value based on whose turn it is
        # White wants to maximize (highest value), Black wants to minimize (lowest value)
        best_value = -float('inf') if board.turn else float('inf')
        best_line = []
        alpha = -float('inf')
        beta = float('inf')
        
        # Store best line and value as instance variables for logging
        self.best_line = []
        self.best_value = 0
        
        self.logger.log_search_start(board, search_depth)
        
        # Start search visualization if enabled
        self.visualizer.start_search(board, search_depth)
        
        # Check if position is in tablebase
        if self.tablebase and self.is_tablebase_position(board):
            if not self.quiet:
                print(f"🔍 Checking tablebase for position with {sum(len(board.pieces(piece_type, color)) for piece_type in chess.PIECE_TYPES for color in [chess.WHITE, chess.BLACK])} pieces")
            tablebase_move = self.get_tablebase_move(board)
            if tablebase_move:
                if not self.quiet:
                    print(f"🎯 Using tablebase move: {board.san(tablebase_move)}")
                return tablebase_move
            else:
                if not self.quiet:
                    print("⚠️  Tablebase lookup failed, using standard search")
        
        # Start timing the search and reset node counter
        start_time = time.time()
        self.search_start_time = start_time
        self.time_budget = time_budget  # Store time budget for minimax function
        self.nodes_searched = 0
        self.search_interrupted = False  # Track if search was interrupted by time budget
        self.repetition_detected = repetition_detected  # Store repetition information
        
        # Phase 1: Perform shallow search for move ordering (no visualization)
        sorted_moves = self._order_moves_with_shallow_search(board)
        
        # Phase 2: Clean transposition table after shallow search
        self._clean_transposition_table_after_shallow_search()
        
        # Evaluate moves in optimal order
        for move in sorted_moves:
            # Record move being considered for visualization
            self.visualizer.record_move_considered(move, board)
            # Check time budget
            if time_budget is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_budget:
                    if not self.quiet:
                        print(f"⏰ Time budget reached ({elapsed_time:.2f}s >= {time_budget:.2f}s), returning current best move")
                    self.search_interrupted = True
                    break
            
            # Get the SAN notation before making the move
            move_san = board.san(move)
            
            # Make the move on the board
            board.push(move)
            
            # Search the resulting position
            # Note: After board.push(move), board.turn has changed to the opponent
            value, line = self._minimax(board, search_depth - 1, alpha, beta, [move_san])
            
            # Undo the move to restore the original board state
            board.pop()
            
            # Update best move based on whose turn it is
            if board.turn:  # White to move: pick highest evaluation
                if value > best_value:
                    best_value = value
                    best_move = move
                    best_line = [move] + line
                    # Update instance variables for logging
                    self.best_line = best_line
                    self.best_value = best_value
                    # Log new best move found during search
                    self.logger.log_new_best_move(move_san, value)
                alpha = max(alpha, value)
            else:  # Black to move: pick lowest evaluation
                if value < best_value:
                    best_value = value
                    best_move = move
                    best_line = [move] + line
                    # Update instance variables for logging
                    self.best_line = best_line
                    self.best_value = best_value
                    # Log new best move found during search
                    self.logger.log_new_best_move(move_san, value)
                beta = min(beta, value)
        
        # Calculate search time and nodes per second
        search_time = time.time() - start_time
        nodes_per_second = self.nodes_searched / search_time if search_time > 0 else 0
        
        # Get transposition table statistics
        tt_stats = self.get_transposition_stats()
        
        self.logger.log_search_completion(search_time, self.nodes_searched, tt_stats)
        
        # Finish search visualization if enabled
        if best_move:
            pv_san = []
            pv_board = board.copy()
            for m in best_line:
                try:
                    pv_san.append(pv_board.san(m))
                    pv_board.push(m)
                except Exception:
                    break
            self.visualizer.finish_search(board.san(best_move), best_value, self.nodes_searched, search_time, pv_san)
            
            # Export visualization to file
            self.move_counter += 1
            viz_file = self.visualizer.export_tree_to_file(self.move_counter)
            if viz_file:
                if not self.quiet:
                    print(f"📄 Search tree exported to: {viz_file}")
        
        # Print the best move found with component evaluation
        if best_move:
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
            
            # Calculate overall evaluation
            overall_eval = final_components['material'] + final_components['position'] + final_components['mobility']
            
            self.logger.log_best_move(board, best_move, best_value, best_line, final_components)
        return best_move

    def _minimax(self, board, depth, alpha, beta, variation=None):
        """
        Minimax search with alpha-beta pruning with transposition table support.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning (best score for maximizing player)
            beta: Beta value for pruning (best score for minimizing player)
            variation: The sequence of moves that led to this position (for debugging)
            
        Returns:
            Tuple of (evaluation, principal_variation)
        """
        # Enter node for visualization
        self.visualizer.enter_node(board, None, depth, alpha, beta, False)
        
        # Check time budget (only if we have a time budget and we're at the root level)
        if hasattr(self, 'search_start_time') and hasattr(self, 'time_budget') and self.time_budget is not None:
            elapsed_time = time.time() - self.search_start_time
            if elapsed_time >= self.time_budget:
                # Return a neutral evaluation to stop the search
                self.visualizer.exit_node(0, "TIMEOUT", [], 0, 0)
                return 0, []
        
        # Count this node
        self.nodes_searched += 1
        
        # Initialize variation if None
        if variation is None:
            variation = []
        
        # Probe transposition table
        tt_score, tt_best_move, tt_node_type = self._probe_transposition(board, depth, alpha, beta)
        
        # If we have a usable transposition table entry, return it
        if tt_score is not None:
            return tt_score, [tt_best_move] if tt_best_move else []
        
        # Leaf node: evaluate position
        if depth == 0:
            # Check if position is in tablebase for perfect evaluation
            if self.tablebase and self.is_tablebase_position(board):
                try:
                    wdl = self.tablebase.get_wdl(board)
                    if wdl is not None:
                        # Convert WDL to evaluation score
                        # WDL: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
                        if wdl == 2:  # Win
                            return 100000, []
                        elif wdl == 1:  # Cursed win
                            return 50000, []
                        elif wdl == 0:  # Draw
                            return 0, []
                        elif wdl == -1:  # Blessed loss
                            return -50000, []
                        elif wdl == -2:  # Loss
                            return -100000, []
                except Exception:
                    # Fall back to quiescence if tablebase lookup fails
                    pass
            
            # Enter quiescence node for visualization
            self.visualizer.enter_node(board, None, 0, alpha, beta, True)
            
            # Perform quiescence search
            result = self._quiescence(board, alpha, beta)
            
            # Exit quiescence node for visualization
            self.visualizer.exit_node(result[0], "QUIESCENCE", result[1] if result[1] else [], 0, 0)
            
            return result
        
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
                eval_score = self.evaluate(board)
            
            # Exit node for visualization
            self.visualizer.exit_node(eval_score, "TERMINAL", [], 0, 0)
            return eval_score, []
        
        # Use optimized move generation and sorting, with TT best move first if available
        sorted_moves = self._get_sorted_moves_optimized(board, tt_best_move)
        
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
                
                # Make move
                board.push(move)
                # Recursively search the resulting position
                eval, line = self._minimax(board, depth - 1, alpha, beta, variation + [move_san])
                # Undo move
                board.pop()
                
                # Apply checkmate distance correction for winning positions
                if abs(eval) >= 50000:  # Likely a checkmate score
                    checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                    if eval > 0:  # White is winning
                        # Calculate distance to mate from the principal variation length
                        # The line contains the moves to mate, so distance = len(line)
                        distance_to_mate = len(line)
                        # Apply correction: prefer shorter mates
                        eval = checkmate_bonus - distance_to_mate
                    else:  # Black is winning
                        # Calculate distance to mate from the principal variation length
                        distance_to_mate = len(line)
                        # Apply correction: prefer shorter mates
                        eval = -(checkmate_bonus - distance_to_mate)
                
                # Apply repetition penalty/bonus based on current position
                if hasattr(self, 'repetition_detected') and self.repetition_detected:
                    repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                    # If we're winning and this move leads to repetition, penalize it
                    # If we're losing and this move leads to repetition, prefer it
                    if eval > 100:  # White is winning
                        eval = repetition_eval  # Prefer draw over winning
                    elif eval < -100:  # White is losing
                        eval = repetition_eval  # Prefer draw over losing
                
                # Update best move if this is better
                if eval > max_eval:
                    max_eval = eval
                    best_line = [move] + line
                    best_move = move
                    # Only print when we find a new best move that's at least 1 point better! 🎯
                    if depth == self.depth - 1:  # Only at top level
                        # Check if this is significantly better (at least 1 point improvement)
                        if eval > max_eval + 1 or max_eval == -float('inf'):
                            variation_str = " -> ".join(variation + [move_san]) if variation else move_san
                            print(f"  🌟 New best! {move_san}: {round(eval, 2)} | Variation: {variation_str}")
                
                # Alpha-beta pruning
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            
            # Store result in transposition table
            if best_move:
                if max_eval <= original_alpha:
                    node_type = UPPER_BOUND
                elif max_eval >= beta:
                    node_type = LOWER_BOUND
                else:
                    node_type = EXACT
                
                self._store_transposition(board, depth, max_eval, best_move, node_type)
                    
            # Exit node for visualization
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
            return max_eval, best_line
            
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
                
                # Make move
                board.push(move)
                # Recursively search the resulting position
                eval, line = self._minimax(board, depth - 1, alpha, beta, variation + [move_san])
                # Undo move
                board.pop()
                
                # Apply checkmate distance correction for winning positions
                if abs(eval) >= 50000:  # Likely a checkmate score
                    checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                    if eval > 0:  # White is winning
                        # Calculate distance to mate from the principal variation length
                        # The line contains the moves to mate, so distance = len(line)
                        distance_to_mate = len(line)
                        # Apply correction: prefer shorter mates
                        eval = checkmate_bonus - distance_to_mate
                    else:  # Black is winning
                        # Calculate distance to mate from the principal variation length
                        distance_to_mate = len(line)
                        # Apply correction: prefer shorter mates
                        eval = -(checkmate_bonus - distance_to_mate)
                
                # Apply repetition penalty/bonus based on current position
                if hasattr(self, 'repetition_detected') and self.repetition_detected:
                    repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                    # If we're winning and this move leads to repetition, penalize it
                    # If we're losing and this move leads to repetition, prefer it
                    if eval < -100:  # Black is winning
                        eval = repetition_eval  # Prefer draw over winning
                    elif eval > 100:  # Black is losing
                        eval = repetition_eval  # Prefer draw over losing
                
                # Update best move if this is better
                if eval < min_eval:
                    min_eval = eval
                    best_line = [move] + line
                    best_move = move
                    # Only print when we find a new best move that's at least 1 point better! 🎯
                    if depth == self.depth - 1:  # Only at top level
                        # Check if this is significantly better (at least 1 point improvement)
                        if eval < min_eval - 1 or min_eval == float('inf'):
                            variation_str = " -> ".join(variation + [move_san]) if variation else move_san
                            print(f"  🌟 New best! {move_san}: {round(eval, 2)} | Variation: {variation_str}")
                
                # Alpha-beta pruning
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            # Store result in transposition table
            if best_move:
                if min_eval <= alpha:
                    node_type = UPPER_BOUND
                elif min_eval >= original_beta:
                    node_type = LOWER_BOUND
                else:
                    node_type = EXACT
                
                self._store_transposition(board, depth, min_eval, best_move, node_type)
                    
            # Exit node for visualization
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
            return min_eval, best_line

    def evaluate(self, board):
        """
        Evaluate the current board position using the evaluation manager.
        
        Args:
            board: Current board state
            
        Returns:
            Evaluation score from White's perspective (rounded to 2 decimal places)
        """
        evaluation = self.evaluation_manager.evaluate(board)
        return round(evaluation, 2)
    
    def evaluate_with_components(self, board):
        """
        Evaluate the current board position with component breakdown.
        
        Args:
            board: Current board state
            
        Returns:
            Dictionary with evaluation components and total score
        """
        return self.evaluation_manager.evaluate_with_components(board)
    
    def evaluate_cached(self, board):
        """
        Evaluate the current board position with caching for repeated positions.
        
        Args:
            board: Current board state
            
        Returns:
            Evaluation score from White's perspective (rounded to 2 decimal places)
        """
        # Use board hash as cache key
        board_hash = hash(board._transposition_key())
        
        if board_hash in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[board_hash]
        
        self.cache_misses += 1
        evaluation = self.evaluation_manager.evaluate(board)
        rounded_evaluation = round(evaluation, 2)
        self.eval_cache[board_hash] = rounded_evaluation
        
        # Limit cache size to prevent memory issues
        cache_size_limit = self.evaluation_manager.evaluator.config.get("cache_size_limit", 10000)
        if len(self.eval_cache) > cache_size_limit:
            # Clear cache if it gets too large
            self.eval_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
        
        return rounded_evaluation
    
    def get_evaluator_info(self):
        """Get information about the current evaluator"""
        return self.evaluation_manager.get_evaluator_info()
    
    def switch_evaluator(self, evaluator_type: str, **kwargs):
        """Switch to a different evaluator"""
        self.evaluation_manager.switch_evaluator(evaluator_type, **kwargs)
    
    def get_evaluation_history(self):
        """Get evaluation history for analysis"""
        return self.evaluation_manager.get_evaluation_history()
    
    def clear_evaluation_history(self):
        """Clear evaluation history"""
        self.evaluation_manager.clear_history()
    
    def get_cache_stats(self):
        """Get evaluation cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self.eval_cache)
        }

    def _quiescence(self, board, alpha, beta, depth=0):
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
            Tuple of (evaluation, principal_variation)
        """
        # Count this node
        self.nodes_searched += 1
        
        # Check for game over conditions first
        if board.is_game_over():
            if board.is_checkmate():
                # Checkmate evaluation
                checkmate_bonus = self.evaluation_manager.evaluator.config.get("checkmate_bonus", 100000)
                if board.turn:  # Black wins (White to move but checkmated)
                    return -checkmate_bonus, []
                else:  # White wins (Black to move but checkmated)
                    return checkmate_bonus, []
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
                # Draw conditions (except repetition)
                draw_value = self.evaluation_manager.evaluator.config.get("draw_value", 0)
                return draw_value, []
            elif board.is_repetition():
                # 3-fold repetition - can be evaluated differently
                repetition_eval = self.evaluation_manager.evaluator.config.get("repetition_evaluation", 0)
                return repetition_eval, []
            else:
                # Other game over conditions
                return self.evaluate(board), []
            
        # Limit quiescence depth to prevent infinite loops
        quiescence_depth_limit = self.evaluation_manager.evaluator.config.get("quiescence_depth_limit", 10)
        if depth > quiescence_depth_limit:
            return self.evaluate(board), []
        
        # Check if position is in tablebase for perfect evaluation
        if self.tablebase and self.is_tablebase_position(board):
            try:
                wdl = self.tablebase.get_wdl(board)
                if wdl is not None:
                    # Convert WDL to evaluation score
                    # WDL: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
                    if wdl == 2:  # Win
                        return 100000, []
                    elif wdl == 1:  # Cursed win
                        return 50000, []
                    elif wdl == 0:  # Draw
                        return 0, []
                    elif wdl == -1:  # Blessed loss
                        return -50000, []
                    elif wdl == -2:  # Loss
                        return -100000, []
            except Exception:
                # Fall back to standard evaluation if tablebase lookup fails
                pass
        
        # Probe transposition table for quiescence search (use depth 0 to indicate quiescence)
        tt_score, tt_best_move, tt_node_type = self._probe_transposition(board, 0, alpha, beta)
        
        # If we have a usable transposition table entry, return it
        if tt_score is not None:
            return tt_score, [tt_best_move] if tt_best_move else []
        
        # Check if position is in check
        is_in_check = board.is_check()
        
        # Evaluate current position (stand pat) - only if not in check
        if not is_in_check:
            stand_pat = self.evaluate_cached(board)
            
            # Alpha-beta pruning at quiescence level
            if board.turn:  # White to move: maximize
                if stand_pat >= beta:
                    return beta, []
                alpha = max(alpha, stand_pat)
            else:  # Black to move: minimize
                if stand_pat <= alpha:
                    return alpha, []
                beta = min(beta, stand_pat)
        
        # Determine which moves to search
        if is_in_check:
            # If in check, search ALL legal moves (the opponent must respond to the check)
            moves_to_search = list(board.legal_moves)
        else:
            # If not in check, search captures and checks for tactical accuracy
            moves_to_search = self._get_quiescence_moves(board)
        if not moves_to_search:
            if is_in_check:
                # If in check and no legal moves, it's checkmate
                return -100000 if board.turn else 100000, []
            else:
                # If not in check and no captures/checks, return stand pat
                stand_pat = self.evaluate_cached(board)
                return stand_pat, []
        
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
            score, line = self._quiescence(board, alpha, beta, depth + 1)
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
        
        # Store result in transposition table for quiescence search
        final_score = alpha if board.turn else beta
        if best_move:
            if board.turn:  # White to move
                if final_score <= original_alpha:
                    node_type = UPPER_BOUND
                elif final_score >= original_beta:
                    node_type = LOWER_BOUND
                else:
                    node_type = EXACT
            else:  # Black to move
                if final_score <= original_alpha:
                    node_type = UPPER_BOUND
                elif final_score >= original_beta:
                    node_type = LOWER_BOUND
                else:
                    node_type = EXACT
            
            self._store_transposition(board, 0, final_score, best_move, node_type)
        
        return (alpha, best_line) if board.turn else (beta, best_line)
    
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
        victim_piece = board.piece_at(move.to_square)
        attacker_piece = board.piece_at(move.from_square)
        
        if victim_piece is None or attacker_piece is None:
            return 0
        
        # Piece values for MVV-LVA (different from evaluation values)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        
        # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
        # Higher values = more promising captures
        return piece_values[victim_piece.piece_type] * 10 - piece_values[attacker_piece.piece_type]
    

    
    def _get_sorted_moves_optimized(self, board, tt_best_move=None):
        """
        Optimized move generation and sorting with transposition table support.
        
        Args:
            board: Current board state
            tt_best_move: Best move from transposition table (if available)
            
        Returns:
            List of moves sorted by priority (TT move first, then captures, then checks, then others)
        """
        # Use generator to avoid creating full list initially
        legal_moves = board.legal_moves
        captures = []
        checks = []
        non_captures = []
        tt_move_found = False
        
        # Single pass to categorize moves
        for move in legal_moves:
            # Check if this is the TT best move
            if tt_best_move and move == tt_best_move:
                tt_move_found = True
                continue  # We'll add it first later
            
            if board.is_capture(move):
                captures.append(move)
            else:
                # Check if move gives check (but not checkmate - that's handled by search)
                board.push(move)
                if board.is_check():
                    checks.append(move)
                else:
                    non_captures.append(move)
                board.pop()
        
        # Pre-calculate capture values for better sorting
        if captures:
            capture_values = []
            for move in captures:
                value = self._get_capture_value(board, move)
                capture_values.append((move, value))
            
            # Sort captures by value (highest first)
            capture_values.sort(key=lambda x: x[1], reverse=True)
            sorted_captures = [move for move, _ in capture_values]
        else:
            sorted_captures = []
        
        # Combine moves: TT move first (if found), then captures, then checks, then other moves
        result = []
        if tt_move_found and tt_best_move:
            result.append(tt_best_move)
        result.extend(sorted_captures)
        result.extend(checks)
        result.extend(non_captures)
        
        return result

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
    
    def _is_endgame_position(self, board):
        """Check if position is an endgame for evaluation purposes"""
        total_pieces = chess.popcount(board.occupied)
        return total_pieces <= 12
    
    def get_tablebase_move(self, board):
        """Get the best move from tablebase if available"""
        try:
            if not self.tablebase:
                if not self.quiet:
                    print("⚠️  No tablebase available")
                return None
            
            # Get tablebase information for current position
            wdl = self.tablebase.get_wdl(board)
            if wdl is None:
                if not self.quiet:
                    print("⚠️  Position not found in tablebase")
                return None
            
            # WDL values: 2 = win, 1 = cursed win, 0 = draw, -1 = blessed loss, -2 = loss
            wdl_names = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
            if not self.quiet:
                print(f"📊 Tablebase: WDL={wdl} ({wdl_names.get(wdl, 'Unknown')})")
            
            # Get DTZ for current position to understand the fastest path
            dtz = self.tablebase.get_dtz(board)
            if dtz is not None and not self.quiet:
                print(f"📊 Tablebase: DTZ={dtz} (Distance To Zero)")
            
            # Find the best move by trying each legal move
            best_move = None
            best_wdl = None
            best_dtz = None
            
            if not self.quiet:
                print(f"🔍 Checking {len(list(board.legal_moves))} legal moves...")
            
            moves_checked = 0
            for move in board.legal_moves:
                # Get SAN notation before pushing the move
                move_san = board.san(move)
                
                board.push(move)
                try:
                    move_wdl = self.tablebase.get_wdl(board)
                    move_dtz = self.tablebase.get_dtz(board)
                    moves_checked += 1
                    
                    if move_wdl is not None:
                        # The WDL value is from the perspective of the side that just moved
                        # We need to find the best move for the side that is about to move
                        
                        # When White is to move (board.turn was True), after making a move, board.turn becomes False (Black to move)
                        # When Black is to move (board.turn was False), after making a move, board.turn becomes True (White to move)
                        
                        if not board.turn:  # White just moved, so Black is to move next
                            # We want to find the best move for White
                            # The WDL value represents what Black can achieve after White's move
                            # White wants to minimize Black's best result (minimize WDL from Black's perspective)
                            # Among moves with the same WDL, choose the one with lowest DTZ (fastest win for White)
                            if (best_wdl is None or 
                                move_wdl < best_wdl or 
                                (move_wdl == best_wdl and move_dtz is not None and 
                                 (best_dtz is None or move_dtz < best_dtz))):
                                best_wdl = move_wdl
                                best_dtz = move_dtz
                                best_move = move
                                print(f"  ✅ {move_san}: WDL={move_wdl} ({wdl_names.get(move_wdl, 'Unknown')}), DTZ={move_dtz} - White's best move")
                        else:  # Black just moved, so White is to move next
                            # We want to find the best move for Black
                            # The WDL value represents what White can achieve after Black's move
                            # Black wants to minimize White's best result (minimize WDL from White's perspective)
                            # Among moves with the same WDL, choose the one with highest DTZ (slowest win for White)
                            if (best_wdl is None or 
                                move_wdl < best_wdl or 
                                (move_wdl == best_wdl and move_dtz is not None and 
                                 (best_dtz is None or move_dtz > best_dtz))):
                                best_wdl = move_wdl
                                best_dtz = move_dtz
                                best_move = move
                                print(f"  ✅ {move_san}: WDL={move_wdl} ({wdl_names.get(move_wdl, 'Unknown')}), DTZ={move_dtz} - Black's best move")
                    else:
                        print(f"  ❌ {move_san}: Not found in tablebase")
                except Exception as e:
                    if not self.quiet:
                        print(f"⚠️  Error checking move {move_san}: {e}")
                board.pop()
            
            if not self.quiet:
                print(f"📊 Checked {moves_checked} moves in tablebase")
            
            if best_move:
                best_move_san = board.san(best_move)
                side_to_move = "White" if board.turn else "Black"
                if not self.quiet:
                    print(f"🎯 Tablebase best move for {side_to_move}: {best_move_san} (WDL={best_wdl} - {wdl_names.get(best_wdl, 'Unknown')}, DTZ={best_dtz})")
            else:
                if not self.quiet:
                    print("⚠️  No best move found in tablebase")
            
            return best_move
            
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  Tablebase error: {e}")
            return None 
    def _get_quiescence_moves(self, board):
        """
        Get moves to search in quiescence, including defensive moves based on configuration.
        
        Args:
            board: Current board state
            
        Returns:
            List of moves to search in quiescence
        """
        moves_to_search = []
        
        # Always include captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        moves_to_search.extend(captures)
        
        # Include checks if configured
        include_checks = self.evaluation_manager.evaluator.config.get("quiescence_include_checks", True)
        if include_checks:
            checks = []
            for move in board.legal_moves:
                if not board.is_capture(move):
                    board.push(move)
                    if board.is_check():
                        checks.append(move)
                    board.pop()
            moves_to_search.extend(checks)
        
        # Include defensive moves based on configuration
        include_queen_defense = self.evaluation_manager.evaluator.config.get("quiescence_include_queen_defense", False)
        include_value_threshold = self.evaluation_manager.evaluator.config.get("quiescence_include_value_threshold", False)
        
        if include_queen_defense or include_value_threshold:
            defensive_moves = self._get_defensive_moves(board, include_queen_defense, include_value_threshold)
            moves_to_search.extend(defensive_moves)
        
        return moves_to_search
    
    def _get_defensive_moves(self, board, include_queen_defense, include_value_threshold):
        """
        Get defensive moves based on configuration options.
        
        Args:
            board: Current board state
            include_queen_defense: Whether to include queen defensive moves
            include_value_threshold: Whether to include value threshold defensive moves
            
        Returns:
            List of defensive moves
        """
        defensive_moves = []
        piece_values = self.evaluation_manager.evaluator.config.get("piece_values", {})
        value_threshold = self.evaluation_manager.evaluator.config.get("quiescence_value_threshold", 500)
        
        for move in board.legal_moves:
            # Skip moves already included (captures and checks)
            if board.is_capture(move) or self._gives_check(board, move):
                continue
            
            # Check if this move is defensive based on configuration
            if self._is_defensive_move(board, move, include_queen_defense, include_value_threshold, piece_values, value_threshold):
                defensive_moves.append(move)
        
        return defensive_moves
    
    def _is_defensive_move(self, board, move, include_queen_defense, include_value_threshold, piece_values, value_threshold):
        """
        Check if a move is defensive based on configuration options.
        
        Args:
            board: Current board state
            move: The move to check
            include_queen_defense: Whether to include queen defensive moves
            include_value_threshold: Whether to include value threshold defensive moves
            piece_values: Dictionary of piece values
            value_threshold: Value threshold for defensive moves
            
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
            piece_symbol = piece.symbol().upper()
            if piece_symbol in piece_values:
                piece_value = piece_values[piece_symbol]
                if piece_value >= value_threshold:
                    return True
        
        return False
    
    def _gives_check(self, board, move):
        """
        Check if a move gives check.
        
        Args:
            board: Current board state
            move: The move to check
            
        Returns:
            True if the move gives check, False otherwise
        """
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        return gives_check

