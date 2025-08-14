import random
import chess
import chess.syzygy
import time
import os
import zlib
try:
    from .evaluation import EvaluationManager, create_evaluator
except ImportError:
    from evaluation import EvaluationManager, create_evaluator

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
    
    def __init__(self, depth=None, evaluator_type="handcrafted", evaluator_config=None, quiet=False):
        self.nodes_searched = 0
        self.search_start_time = 0
        self.quiet = quiet  # Control debug output
        
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

    def get_move(self, board):
        """
        Find the best move for the current side to move.
        
        Args:
            board: Current chess board state
            
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
        
        if not self.quiet:
            print(f"\n🤔 Engine thinking (depth {search_depth})...")
            print(f"🎭 Current side to move: {'White' if board.turn else 'Black'}")
        
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
        self.nodes_searched = 0
        
        # Evaluate all legal moves
        for move in board.legal_moves:
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
                alpha = max(alpha, value)
            else:  # Black to move: pick lowest evaluation
                if value < best_value:
                    best_value = value
                    best_move = move
                    best_line = [move] + line
                beta = min(beta, value)
        
        # Calculate search time and nodes per second
        search_time = time.time() - start_time
        nodes_per_second = self.nodes_searched / search_time if search_time > 0 else 0
        
        # Get transposition table statistics
        tt_stats = self.get_transposition_stats()
        
        if not self.quiet:
            print(f"⏱️ Search completed in {search_time:.2f}s")
            print(f"🔄 TT: {tt_stats['hits']}/{tt_stats['total_probes']} hits ({tt_stats['hit_rate']:.1f}%) | Cutoffs: {tt_stats['cutoffs']} ({tt_stats['cutoff_rate']:.1f}%)")
        
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
            
            if not self.quiet:
                print(f"🏆 Best: {pv_san[0]} ({best_value:.1f}) | PV: {' '.join(pv_san)} | Speed: {nodes_per_second:.0f} nodes/s")
                print(f"📊 (Material: {final_components['material']}, Position: {final_components['position']}, Mobility: {final_components['mobility']})")
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
            
            return self._quiescence(board, alpha, beta)
        
        # Base case: game over
        if board.is_game_over():
            return self.evaluate(board), []
        
        # Use optimized move generation and sorting, with TT best move first if available
        sorted_moves = self._get_sorted_moves_optimized(board, tt_best_move)
        
        # White's turn: maximize evaluation
        if board.turn:
            max_eval = -float('inf')
            best_line = []
            best_move = None
            original_alpha = alpha
            
            for move in sorted_moves:
                # Get SAN notation before making the move
                move_san = board.san(move)
                
                # Make move
                board.push(move)
                # Recursively search the resulting position
                eval, line = self._minimax(board, depth - 1, alpha, beta, variation + [move_san])
                # Undo move
                board.pop()
                
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
                    
            return max_eval, best_line
            
        # Black's turn: minimize evaluation
        else:
            min_eval = float('inf')
            best_line = []
            best_move = None
            original_beta = beta
            
            for move in sorted_moves:
                # Get SAN notation before making the move
                move_san = board.san(move)
                
                # Make move
                board.push(move)
                # Recursively search the resulting position
                eval, line = self._minimax(board, depth - 1, alpha, beta, variation + [move_san])
                # Undo move
                board.pop()
                
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
            captures = []
            checks = []
            
            # Check if we should include checks in quiescence search
            include_checks = self.evaluation_manager.evaluator.config.get("quiescence_include_checks", True)
            
            for move in board.legal_moves:
                if board.is_capture(move):
                    captures.append(move)
                elif include_checks:
                    # Check if move gives check
                    board.push(move)
                    if board.is_check():
                        checks.append(move)
                    board.pop()
            
            # Combine captures first, then checks
            moves_to_search = captures + checks
        
        # If no moves to search, return stand pat evaluation
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