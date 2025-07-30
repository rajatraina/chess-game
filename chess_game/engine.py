import random
import chess
import chess.syzygy
import time
import os
import sys

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
    
    def __init__(self, depth=4, max_memory_mb=100):
        self.depth = depth
        self.nodes_searched = 0
        self.search_start_time = 0
        
        # Transposition table settings
        self.max_memory_mb = max_memory_mb
        self.transposition_table = {}
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_size = 0
        self.tt_memory_usage = 0
        
        # Initialize Zobrist hash table
        self.zobrist_table = self._init_zobrist_table()
        
        # Initialize Syzygy tablebase if available
        self.tablebase = None
        self.init_tablebase()
    
    def _init_zobrist_table(self):
        """Initialize Zobrist hash table for efficient position hashing"""
        import random
        
        # Create a random number generator with fixed seed for reproducibility
        rng = random.Random(42)
        
        # Initialize Zobrist table
        # 64 squares × 12 piece types (6 pieces × 2 colors) + 1 for en passant + 1 for castling
        zobrist_table = {}
        
        # Piece hashes: 64 squares × 12 piece types
        for square in range(64):
            for piece_type in chess.PIECE_TYPES:
                for color in [chess.WHITE, chess.BLACK]:
                    piece = chess.Piece(piece_type, color)
                    zobrist_table[(square, piece)] = rng.getrandbits(64)
        
        # En passant square hash
        for square in range(64):
            zobrist_table[('en_passant', square)] = rng.getrandbits(64)
        
        # Castling rights hash
        castling_rights = [chess.BB_A1, chess.BB_H1, chess.BB_A8, chess.BB_H8]
        for castling_right in castling_rights:
            zobrist_table[('castling', castling_right)] = rng.getrandbits(64)
        
        # Side to move hash
        zobrist_table['white_to_move'] = rng.getrandbits(64)
        
        return zobrist_table
    
    def _get_zobrist_hash(self, board):
        """Calculate Zobrist hash for the current board position"""
        hash_value = 0
        
        # Hash pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                hash_value ^= self.zobrist_table[(square, piece)]
        
        # Hash en passant square
        if board.ep_square is not None:
            hash_value ^= self.zobrist_table[('en_passant', board.ep_square)]
        
        # Hash castling rights
        castling_rights = [chess.BB_A1, chess.BB_H1, chess.BB_A8, chess.BB_H8]
        for castling_right in castling_rights:
            if board.has_castling_rights(castling_right):
                hash_value ^= self.zobrist_table[('castling', castling_right)]
        
        # Hash side to move
        if board.turn:
            hash_value ^= self.zobrist_table['white_to_move']
        
        return hash_value
    
    def _get_tt_key(self, board, depth, alpha, beta):
        """Generate transposition table key"""
        hash_value = self._get_zobrist_hash(board)
        return (hash_value, depth, alpha, beta)
    
    def _store_tt_entry(self, board, depth, alpha, beta, value, best_move, node_type):
        """Store entry in transposition table"""
        key = self._get_tt_key(board, depth, alpha, beta)
        
        # Estimate memory usage for this entry
        entry_size = sys.getsizeof(key) + sys.getsizeof(value) + sys.getsizeof(best_move) + sys.getsizeof(node_type) + 50  # overhead
        
        # Check if we need to clear some entries due to memory limit
        if self.tt_memory_usage + entry_size > self.max_memory_mb * 1024 * 1024:
            self._clear_tt_table()
        
        # Store the entry
        self.transposition_table[key] = {
            'value': value,
            'best_move': best_move,
            'node_type': node_type,  # 'exact', 'alpha', 'beta'
            'depth': depth
        }
        
        self.tt_size = len(self.transposition_table)
        self.tt_memory_usage += entry_size
    
    def _probe_tt_entry(self, board, depth, alpha, beta):
        """Probe transposition table for existing entry"""
        key = self._get_tt_key(board, depth, alpha, beta)
        
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            
            # Only use entry if it's from a search of sufficient depth
            if entry['depth'] >= depth:
                self.tt_hits += 1
                return entry['value'], entry['best_move'], entry['node_type']
        
        self.tt_misses += 1
        return None, None, None
    
    def _clear_tt_table(self):
        """Clear transposition table to free memory"""
        self.transposition_table.clear()
        self.tt_size = 0
        self.tt_memory_usage = 0
        print(f"🧹 Cleared transposition table (memory limit reached)")
    
    def get_tt_stats(self):
        """Get transposition table statistics"""
        hit_rate = self.tt_hits / (self.tt_hits + self.tt_misses) if (self.tt_hits + self.tt_misses) > 0 else 0
        memory_mb = self.tt_memory_usage / (1024 * 1024)
        
        return {
            'hits': self.tt_hits,
            'misses': self.tt_misses,
            'hit_rate': hit_rate,
            'size': self.tt_size,
            'memory_mb': memory_mb,
            'max_memory_mb': self.max_memory_mb
        }
    
    def clear_tt_table(self):
        """Clear transposition table and reset statistics"""
        self.transposition_table.clear()
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_size = 0
        self.tt_memory_usage = 0
        print(f"🧹 Cleared transposition table and reset statistics")
    
    def set_tt_memory_limit(self, max_memory_mb):
        """Set the maximum memory limit for transposition table"""
        self.max_memory_mb = max_memory_mb
        print(f"💾 Set transposition table memory limit to {max_memory_mb}MB")
    
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
        import copy
        best_move = None
        # Initialize best_value based on whose turn it is
        # White wants to maximize (highest value), Black wants to minimize (lowest value)
        best_value = -float('inf') if board.turn else float('inf')
        best_line = []
        alpha = -float('inf')
        beta = float('inf')
        
        print(f"\n🤔 Engine thinking (depth {self.depth})...")
        print(f"🎭 Current side to move: {'White' if board.turn else 'Black'}")
        
        # Check if position is in tablebase
        if self.tablebase and self.is_endgame_position(board):
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
            value, line = self._minimax(board, self.depth - 1, alpha, beta, [move_san])
            
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
        tt_stats = self.get_tt_stats()
        
        print(f"⏱️ Search completed in {search_time:.2f}s")
        print(f"🧠 TT: {tt_stats['hits']} hits, {tt_stats['misses']} misses ({tt_stats['hit_rate']:.1%} hit rate)")
        print(f"💾 TT: {tt_stats['size']} entries, {tt_stats['memory_mb']:.1f}MB / {tt_stats['max_memory_mb']}MB")
        
        # Print the best move found
        if best_move:
            pv_board = board.copy()
            pv_san = []
            for m in best_line:
                try:
                    pv_san.append(pv_board.san(m))
                    pv_board.push(m)
                except Exception:
                    break
            print(f"🏆 Best: {pv_san[0]} ({best_value}) | PV: {' '.join(pv_san)} | Speed: {nodes_per_second:.0f} nodes/s")
        return best_move

    def _minimax(self, board, depth, alpha, beta, variation=None):
        """
        Minimax search with alpha-beta pruning and transposition table.
        
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
        
        # Check transposition table
        tt_value, tt_move, tt_node_type = self._probe_tt_entry(board, depth, alpha, beta)
        if tt_value is not None:
            # Use transposition table entry
            if tt_node_type == 'exact':
                return tt_value, [tt_move] if tt_move else []
            elif tt_node_type == 'alpha' and tt_value <= alpha:
                return alpha, [tt_move] if tt_move else []
            elif tt_node_type == 'beta' and tt_value >= beta:
                return beta, [tt_move] if tt_move else []
            
        # Leaf node: evaluate position
        if depth == 0:
            eval = self.evaluate(board)
            return self._quiescence(board, alpha, beta)
        
        # Base case: game over
        if board.is_game_over():
            eval = self.evaluate(board)
            return self._quiescence(board, alpha, beta)
        
        # Sort moves to prioritize transposition table move first, then captures
        legal_moves = list(board.legal_moves)
        
        # Put transposition table move first if available
        sorted_moves = []
        if tt_move and tt_move in legal_moves:
            sorted_moves.append(tt_move)
            legal_moves.remove(tt_move)
        
        # Then prioritize captures
        captures = [move for move in legal_moves if board.is_capture(move)]
        non_captures = [move for move in legal_moves if not board.is_capture(move)]
        
        # Sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        captures.sort(key=lambda move: self._get_capture_value(board, move), reverse=True)
        
        # Combine: TT move first, then captures, then other moves
        sorted_moves.extend(captures + non_captures)
        
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
                            print(f"  🌟 New best! {move_san}: {eval} | Variation: {variation_str}")
                
                # Alpha-beta pruning
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            
            # Store result in transposition table
            node_type = 'exact'
            if max_eval <= alpha:
                node_type = 'alpha'
            elif max_eval >= beta:
                node_type = 'beta'
            
            self._store_tt_entry(board, depth, alpha, beta, max_eval, best_line[0] if best_line else None, node_type)
                    
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
                            print(f"  🌟 New best! {move_san}: {eval} | Variation: {variation_str}")
                
                # Alpha-beta pruning
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            # Store result in transposition table
            node_type = 'exact'
            if min_eval <= alpha:
                node_type = 'alpha'
            elif min_eval >= beta:
                node_type = 'beta'
            
            self._store_tt_entry(board, depth, alpha, beta, min_eval, best_line[0] if best_line else None, node_type)
                    
            return min_eval, best_line

    def evaluate(self, board):
        """
        Evaluate the current board position.
        
        Evaluation is always from White's perspective:
        - Positive = good for White
        - Negative = good for Black
        
        Args:
            board: Current board state
            
        Returns:
            Evaluation score
        """
        # Material values (from White's perspective)
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-square tables (positional bonuses/penalties)
        # Higher values = better squares for that piece
        pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            2, 4, 4, -8, -8, 4, 4, 2,
            2, -2, -4, 0, 0, -4, -2, 2,
            0, 0, 0, 8, 8, 0, 0, 0,
            2, 2, 4, 12, 12, 4, 2, 2,
            4, 4, 8, 15, 15, 8, 4, 4,
            25, 25, 25, 25, 25, 25, 25, 25,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        knight_table = [
            -25, -20, -15, -15, -15, -15, -20, -25,
            -20, -10, 0, 0, 0, 0, -10, -20,
            -15, 0, 5, 8, 8, 5, 0, -15,
            -15, 2, 8, 10, 10, 8, 2, -15,
            -15, 0, 8, 10, 10, 8, 0, -15,
            -15, 2, 5, 8, 8, 5, 2, -15,
            -20, -10, 0, 2, 2, 0, -10, -20,
            -25, -20, -15, -15, -15, -15, -20, -25
        ]
        bishop_table = [
            -10, -5, -5, -5, -5, -5, -5, -10,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 2, 5, 5, 2, 0, -5,
            -5, 2, 2, 5, 5, 2, 2, -5,
            -5, 0, 5, 5, 5, 5, 0, -5,
            -5, 5, 5, 5, 5, 5, 5, -5,
            -5, 2, 0, 0, 0, 0, 2, -5,
            -10, -5, -5, -5, -5, -5, -5, -10
        ]
        rook_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            2, 4, 4, 4, 4, 4, 4, 2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            -2, 0, 0, 0, 0, 0, 0, -2,
            0, 0, 0, 2, 2, 0, 0, 0
        ]
        queen_table = [
            -10, -5, -5, -2, -2, -5, -5, -10,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 2, 2, 2, 2, 0, -5,
            -2, 0, 2, 2, 2, 2, 0, -2,
            0, 0, 2, 2, 2, 2, 0, -2,
            -5, 2, 2, 2, 2, 2, 0, -5,
            -5, 0, 2, 0, 0, 0, 0, -5,
            -10, -5, -5, -2, -2, -5, -5, -10
        ]
        king_table = [
            -15, -20, -20, -25, -25, -20, -20, -15,
            -15, -20, -20, -25, -25, -20, -20, -15,
            -15, -20, -20, -25, -25, -20, -20, -15,
            -15, -20, -20, -25, -25, -20, -20, -15,
            -10, -15, -15, -20, -20, -15, -15, -10,
            -5, -10, -10, -10, -10, -10, -10, -5,
            10, 10, 0, 0, 0, 0, 10, 10,
            10, 15, 5, 0, 0, 5, 15, 10
        ]
        pst = {
            chess.PAWN: pawn_table,
            chess.KNIGHT: knight_table,
            chess.BISHOP: bishop_table,
            chess.ROOK: rook_table,
            chess.QUEEN: queen_table,
            chess.KING: king_table
        }
        
        # Calculate material value (primary factor)
        material_value = 0
        for piece_type in piece_values:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            material_value += white_count * piece_values[piece_type]
            material_value -= black_count * piece_values[piece_type]
        
        # Calculate positional value (secondary factor)
        positional_value = 0
        for piece_type in piece_values:
            # Add positional bonuses for White pieces
            for square in board.pieces(piece_type, chess.WHITE):
                positional_value += pst[piece_type][square]
            # Subtract positional bonuses for Black pieces (mirror the board)
            for square in board.pieces(piece_type, chess.BLACK):
                positional_value -= pst[piece_type][chess.square_mirror(square)]
        
        # Combine material and positional values
        # Positional value has small weight (0.1) compared to material
        value = material_value + 0.1 * positional_value
        
        # Add checkmate bonus/penalty
        if board.is_checkmate():
            if board.turn:  # White is checkmated
                value -= 100000
            else:  # Black is checkmated
                value += 100000
        
        # Three-fold repetition detection (draw)
        if board.is_repetition(3):
            value = 0  # Draw by repetition
        
        # Fifty-move rule detection
        # If 50 moves have passed without pawn moves or captures, it's a draw
        if board.halfmove_clock >= 100:  # 50 moves = 100 half-moves
            value = 0  # Draw by fifty-move rule
            
        return value

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
        
        # Limit quiescence depth to prevent infinite loops
        if depth > 10:
            return self.evaluate(board), []
            
        # Evaluate current position (stand pat)
        stand_pat = self.evaluate(board)
        
        # Alpha-beta pruning at quiescence level
        if board.turn:  # White to move: maximize
            if stand_pat >= beta:
                return beta, []
            alpha = max(alpha, stand_pat)
        else:  # Black to move: minimize
            if stand_pat <= alpha:
                return alpha, []
            beta = min(beta, stand_pat)
        
        # Only search captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        
        # Sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        # This improves move ordering and pruning efficiency
        captures.sort(key=lambda move: self._get_capture_value(board, move), reverse=True)
        
        best_line = []
        for move in captures:
            # Make the capture
            board.push(move)
            # Recursively search the resulting position
            score, line = self._quiescence(board, alpha, beta, depth + 1)
            # Undo the capture
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

    def is_endgame_position(self, board):
        """Check if position is suitable for tablebase lookup"""
        # Count pieces
        piece_count = 0
        for piece_type in chess.PIECE_TYPES:
            piece_count += len(board.pieces(piece_type, chess.WHITE))
            piece_count += len(board.pieces(piece_type, chess.BLACK))
        
        # Tablebases work best with 7 or fewer pieces
        return piece_count <= 7
    
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