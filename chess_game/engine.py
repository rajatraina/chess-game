import random
import chess
import chess.syzygy
import time
import os
try:
    from .evaluation import EvaluationManager, create_evaluator
except ImportError:
    from evaluation import EvaluationManager, create_evaluator

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
    
    def __init__(self, depth=6, evaluator_type="handcrafted", evaluator_config=None):
        self.depth = depth
        self.nodes_searched = 0
        self.search_start_time = 0
        
        # Initialize evaluation system with config file by default
        if evaluator_config is None:
            evaluator_config = {"config_file": "chess_game/evaluation_config.json"}
        
        self.evaluation_manager = EvaluationManager(evaluator_type, **(evaluator_config or {}))
        
        # Initialize Syzygy tablebase if available
        self.tablebase = None
        self.init_tablebase()
        
        # Evaluation cache for repeated positions
        self.eval_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
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
        # Check if this is an endgame position for deeper search
        is_endgame = self._is_endgame_position(board)
        search_depth = self.depth
        
        if is_endgame:
            endgame_depth = self.evaluation_manager.evaluator.config.get("endgame_search_depth", 6)
            search_depth = max(self.depth, endgame_depth)
            print(f"🎯 Endgame detected - using depth {search_depth}")
        best_move = None
        # Initialize best_value based on whose turn it is
        # White wants to maximize (highest value), Black wants to minimize (lowest value)
        best_value = -float('inf') if board.turn else float('inf')
        best_line = []
        alpha = -float('inf')
        beta = float('inf')
        
        print(f"\n🤔 Engine thinking (depth {search_depth})...")
        print(f"🎭 Current side to move: {'White' if board.turn else 'Black'}")
        
        # Check if position is in tablebase
        if self.tablebase and self.is_tablebase_position(board):
            print(f"🔍 Checking tablebase for position with {sum(len(board.pieces(piece_type, color)) for piece_type in chess.PIECE_TYPES for color in [chess.WHITE, chess.BLACK])} pieces")
            tablebase_move = self.get_tablebase_move(board)
            if tablebase_move:
                print(f"🎯 Using tablebase move: {board.san(tablebase_move)}")
                return tablebase_move
            else:
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
        
        print(f"⏱️ Search completed in {search_time:.2f}s")
        
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
            
            print(f"🏆 Best: {pv_san[0]} ({best_value:.1f}) | PV: {' '.join(pv_san)} | Speed: {nodes_per_second:.0f} nodes/s")
            print(f"📊 (Material: {final_components['material']}, Position: {final_components['position']}, Mobility: {final_components['mobility']})")
        return best_move

    def _minimax(self, board, depth, alpha, beta, variation=None):
        """
        Minimax search with alpha-beta pruning.
        
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
        
        # Leaf node: evaluate position
        if depth == 0:
            return self._quiescence(board, alpha, beta)
        
        # Base case: game over
        if board.is_game_over():
            return self.evaluate(board), []
        
        # Use optimized move generation and sorting
        sorted_moves = self._get_sorted_moves_optimized(board)
        
        # White's turn: maximize evaluation
        if board.turn:
            max_eval = -float('inf')
            best_line = []
            
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
                    
            return max_eval, best_line
            
        # Black's turn: minimize evaluation
        else:
            min_eval = float('inf')
            best_line = []
            
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
        Quiescence search - continues searching only captures to avoid horizon effect.
        
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
            
        # Evaluate current position (stand pat)
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
        
        # Search only captures for efficiency
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        
        # Sort captures by MVV-LVA for better pruning
        if captures:
            capture_values = [(move, self._get_capture_value(board, move)) for move in captures]
            capture_values.sort(key=lambda x: x[1], reverse=True)
            captures = [move for move, _ in capture_values]
        
        moves_to_search = captures
        
        best_line = []
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
                if alpha >= beta:
                    break  # Beta cutoff
            else:  # Black to move: minimize
                if score < beta:
                    beta = score
                    best_line = [move] + line
                if beta <= alpha:
                    break  # Alpha cutoff
        
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
    
    def _get_sorted_moves_optimized(self, board):
        """
        Optimized move generation and sorting.
        
        Args:
            board: Current board state
            
        Returns:
            List of moves sorted by priority (captures first, then checks, then others)
        """
        # Use generator to avoid creating full list initially
        legal_moves = board.legal_moves
        captures = []
        checks = []
        non_captures = []
        
        # Single pass to categorize moves
        for move in legal_moves:
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
        
        # Combine captures first, then checks, then other moves
        return sorted_captures + checks + non_captures

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
                print("⚠️  No tablebase available")
                return None
            
            # Get tablebase information for current position
            wdl = self.tablebase.get_wdl(board)
            if wdl is None:
                print("⚠️  Position not found in tablebase")
                return None
            
            # WDL values: 2 = win, 1 = cursed win, 0 = draw, -1 = blessed loss, -2 = loss
            wdl_names = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
            print(f"📊 Tablebase: WDL={wdl} ({wdl_names.get(wdl, 'Unknown')})")
            
            # Get DTZ for current position to understand the fastest path
            dtz = self.tablebase.get_dtz(board)
            if dtz is not None:
                print(f"📊 Tablebase: DTZ={dtz} (Distance To Zero)")
            
            # Find the best move by trying each legal move
            best_move = None
            best_wdl = None
            best_dtz = None
            
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
                    print(f"⚠️  Error checking move {move_san}: {e}")
                board.pop()
            
            print(f"📊 Checked {moves_checked} moves in tablebase")
            
            if best_move:
                best_move_san = board.san(best_move)
                side_to_move = "White" if board.turn else "Black"
                print(f"🎯 Tablebase best move for {side_to_move}: {best_move_san} (WDL={best_wdl} - {wdl_names.get(best_wdl, 'Unknown')}, DTZ={best_dtz})")
            else:
                print("⚠️  No best move found in tablebase")
            
            return best_move
            
        except Exception as e:
            print(f"⚠️  Tablebase error: {e}")
            return None 