#!/usr/bin/env python3
"""
Search tree visualizer for debugging and understanding engine decision-making.
Can be enabled via configuration without impacting search performance.
"""

import chess
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SearchNode:
    """Represents a node in the search tree."""
    move: Optional[str]  # SAN move or None for root
    depth: int
    alpha: float
    beta: float
    evaluation: float
    node_type: str  # "EXACT", "LOWER_BOUND", "UPPER_BOUND", "PV"
    children: List['SearchNode']
    principal_variation: List[str]
    nodes_searched: int
    time_taken: float
    tt_hit: bool
    tt_depth: int
    quiescence: bool
    evaluation_components: Optional[Dict[str, float]] = None
    move_order: Optional[List[str]] = None
    best_move: Optional[str] = None
    beta_cutoff: bool = False
    cutoff_move: Optional[str] = None
    move_consideration_order: Optional[List[str]] = None  # Order moves were actually considered

class SearchTreeVisualizer:
    """Visualizes the search tree for debugging purposes."""
    
    def __init__(self, enabled: bool = False, target_fen: Optional[str] = None):
        self.enabled = enabled
        self.target_fen = target_fen
        self.search_tree: Optional[SearchNode] = None
        self.current_path: List[SearchNode] = []
        self.node_counter = 0
        
    def should_visualize(self, board: chess.Board) -> bool:
        """Check if we should visualize this position."""
        if not self.enabled:
            return False
        if self.target_fen is None:
            return True
        return board.fen().split()[0] == self.target_fen.split()[0]  # Compare position part only
    
    def start_search(self, board: chess.Board, depth: int):
        """Start visualizing a new search."""
        if not self.should_visualize(board):
            return
            
        self.search_tree = SearchNode(
            move=None,
            depth=depth,
            alpha=-float('inf'),
            beta=float('inf'),
            evaluation=0.0,
            node_type="ROOT",
            children=[],
            principal_variation=[],
            nodes_searched=0,
            time_taken=0.0,
            tt_hit=False,
            tt_depth=0,
            quiescence=False
        )
        self.current_path = [self.search_tree]
        self.node_counter = 0
    
    def enter_node(self, board: chess.Board, move: Optional[chess.Move], depth: int, 
                   alpha: float, beta: float, quiescence: bool = False):
        """Enter a new node in the search tree."""
        if not self.should_visualize(board) or self.search_tree is None:
            return
            
        # Safely convert move to SAN
        move_san = None
        if move:
            try:
                move_san = board.san(move)
            except Exception:
                move_san = move.uci()  # Fallback to UCI notation
        
        node = SearchNode(
            move=move_san,
            depth=depth,
            alpha=alpha,
            beta=beta,
            evaluation=0.0,
            node_type="UNKNOWN",
            children=[],
            principal_variation=[],
            nodes_searched=0,
            time_taken=0.0,
            tt_hit=False,
            tt_depth=0,
            quiescence=quiescence,
            move_consideration_order=[]
        )
        
        # Add to current path
        if self.current_path:
            self.current_path[-1].children.append(node)
        self.current_path.append(node)
        self.node_counter += 1
    
    def record_move_considered(self, move: chess.Move, board: chess.Board):
        """Record a move being considered in the current node."""
        if not self.enabled or not self.current_path:
            return
            
        # Safely convert move to SAN
        try:
            move_san = board.san(move)
        except Exception:
            move_san = move.uci()  # Fallback to UCI notation
        
        if self.current_path[-1].move_consideration_order is None:
            self.current_path[-1].move_consideration_order = []
        self.current_path[-1].move_consideration_order.append(move_san)
    
    def exit_node(self, evaluation: float, node_type: str, principal_variation: List[str],
                  nodes_searched: int, time_taken: float, tt_hit: bool = False, 
                  tt_depth: int = 0, evaluation_components: Optional[Dict[str, float]] = None,
                  move_order: Optional[List[str]] = None, best_move: Optional[str] = None,
                  beta_cutoff: bool = False, cutoff_move: Optional[str] = None):
        """Exit a node and record its results."""
        if not self.enabled or not self.current_path:
            return
            
        node = self.current_path.pop()
        node.evaluation = evaluation
        node.node_type = node_type
        node.principal_variation = principal_variation
        node.nodes_searched = nodes_searched
        node.time_taken = time_taken
        node.tt_hit = tt_hit
        node.tt_depth = tt_depth
        node.evaluation_components = evaluation_components
        node.move_order = move_order
        node.best_move = best_move
        node.beta_cutoff = beta_cutoff
        node.cutoff_move = cutoff_move
    
    def record_tt_hit(self, depth: int):
        """Record a transposition table hit."""
        if self.enabled and self.current_path:
            self.current_path[-1].tt_hit = True
            self.current_path[-1].tt_depth = depth
    
    def record_move_order(self, moves: List[str]):
        """Record the move ordering for this node."""
        if self.enabled and self.current_path:
            self.current_path[-1].move_order = moves
    
    def record_beta_cutoff(self, cutoff_move: str):
        """Record a beta cutoff."""
        if self.enabled and self.current_path:
            self.current_path[-1].beta_cutoff = True
            self.current_path[-1].cutoff_move = cutoff_move
    
    def finish_search(self, best_move: str, best_evaluation: float, total_nodes: int, 
                     total_time: float, final_pv: List[str]):
        """Finish the search and record final results."""
        if not self.enabled or self.search_tree is None:
            return
            
        self.search_tree.best_move = best_move
        self.search_tree.evaluation = best_evaluation
        self.search_tree.nodes_searched = total_nodes
        self.search_tree.time_taken = total_time
        self.search_tree.principal_variation = final_pv
    
    def export_tree_to_file(self, move_number: int = 1) -> str:
        """Export the search tree to a text file in the viz/ directory."""
        if not self.enabled or self.search_tree is None:
            return ""
        
        # Create viz directory if it doesn't exist
        os.makedirs("viz", exist_ok=True)
        
        # Generate filename based on move number
        filename = f"viz/MOVE-{move_number}.txt"
        
        # Write tree to file
        with open(filename, 'w') as f:
            f.write("🌳 Search Tree Visualization\n")
            f.write("=" * 60 + "\n")
            f.write(f"Target FEN: {self.target_fen or 'All positions'}\n")
            f.write(f"Total nodes: {self.node_counter}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("\n")
            
            # Write tree content
            self._write_node_to_file(self.search_tree, f, 0, 10, 5)
        
        return filename
    
    def _write_node_to_file(self, node: SearchNode, file, depth: int, max_depth: int, max_children: int):
        """Recursively write a node and its children to a file."""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        move_str = node.move if node.move else "ROOT"
        
        # Node header
        file.write(f"{indent}📊 {move_str} (d={node.depth}, α={node.alpha:.1f}, β={node.beta:.1f})\n")
        
        # Evaluation info
        eval_str = f"eval={node.evaluation:.1f}"
        if node.evaluation_components:
            comps = [f"{k}={v:.1f}" for k, v in node.evaluation_components.items()]
            eval_str += f" [{', '.join(comps)}]"
        file.write(f"{indent}   {eval_str}\n")
        
        # Node type and stats
        stats = []
        if node.tt_hit:
            stats.append(f"TT(d={node.tt_depth})")
        if node.quiescence:
            stats.append("QS")
        if node.beta_cutoff:
            stats.append(f"β-cut({node.cutoff_move})")
        if node.nodes_searched > 0:
            stats.append(f"nodes={node.nodes_searched}")
        if node.time_taken > 0:
            stats.append(f"time={node.time_taken:.3f}s")
        
        if stats:
            file.write(f"{indent}   [{', '.join(stats)}]\n")
        
        # Move consideration order (actual order moves were considered)
        if node.move_consideration_order:
            moves_str = " ".join(node.move_consideration_order[:10])
            if len(node.move_consideration_order) > 10:
                moves_str += " ..."
            file.write(f"{indent}   Moves considered: {moves_str}\n")
        
        # Move order (if available)
        if node.move_order:
            moves_str = " ".join(node.move_order[:5])
            if len(node.move_order) > 5:
                moves_str += " ..."
            file.write(f"{indent}   Move order: {moves_str}\n")
        
        # Principal variation
        if node.principal_variation:
            # Convert moves to strings if they're Move objects
            pv_moves = []
            for move in node.principal_variation[:3]:
                if isinstance(move, str):
                    pv_moves.append(move)
                else:
                    pv_moves.append(move.uci())  # Convert Move to UCI string
            pv_str = " ".join(pv_moves)
            if len(node.principal_variation) > 3:
                pv_str += " ..."
            file.write(f"{indent}   PV: {pv_str}\n")
        
        file.write("\n")
        
        # Children
        if depth < max_depth:
            children_to_show = node.children[:max_children]
            for child in children_to_show:
                self._write_node_to_file(child, file, depth + 1, max_depth, max_children)
            
            if len(node.children) > max_children:
                file.write(f"{indent}   ... and {len(node.children) - max_children} more children\n")
                file.write("\n")
    
    def print_tree(self, max_depth: int = 3, max_children: int = 5):
        """Print a text representation of the search tree."""
        if not self.enabled or self.search_tree is None:
            return
        
        print("\n🌳 Search Tree Visualization")
        print("=" * 60)
        print(f"Target FEN: {self.target_fen or 'All positions'}")
        print(f"Total nodes: {self.node_counter}")
        print()
        
        self._print_node(self.search_tree, 0, max_depth, max_children)
    
    def _print_node(self, node: SearchNode, depth: int, max_depth: int, max_children: int):
        """Recursively print a node and its children."""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        move_str = node.move if node.move else "ROOT"
        
        # Node header
        print(f"{indent}📊 {move_str} (d={node.depth}, α={node.alpha:.1f}, β={node.beta:.1f})")
        
        # Evaluation info
        eval_str = f"eval={node.evaluation:.1f}"
        if node.evaluation_components:
            comps = [f"{k}={v:.1f}" for k, v in node.evaluation_components.items()]
            eval_str += f" [{', '.join(comps)}]"
        print(f"{indent}   {eval_str}")
        
        # Node type and stats
        stats = []
        if node.tt_hit:
            stats.append(f"TT(d={node.tt_depth})")
        if node.quiescence:
            stats.append("QS")
        if node.beta_cutoff:
            stats.append(f"β-cut({node.cutoff_move})")
        if node.nodes_searched > 0:
            stats.append(f"nodes={node.nodes_searched}")
        if node.time_taken > 0:
            stats.append(f"time={node.time_taken:.3f}s")
        
        if stats:
            print(f"{indent}   [{', '.join(stats)}]")
        
        # Move consideration order (actual order moves were considered)
        if node.move_consideration_order:
            moves_str = " ".join(node.move_consideration_order[:10])
            if len(node.move_consideration_order) > 10:
                moves_str += " ..."
            print(f"{indent}   Moves considered: {moves_str}")
        
        # Move order (if available)
        if node.move_order:
            moves_str = " ".join(node.move_order[:5])
            if len(node.move_order) > 5:
                moves_str += " ..."
            print(f"{indent}   Move order: {moves_str}")
        
        # Principal variation
        if node.principal_variation:
            # Convert moves to strings if they're Move objects
            pv_moves = []
            for move in node.principal_variation[:3]:
                if isinstance(move, str):
                    pv_moves.append(move)
                else:
                    pv_moves.append(move.uci())  # Convert Move to UCI string
            pv_str = " ".join(pv_moves)
            if len(node.principal_variation) > 3:
                pv_str += " ..."
            print(f"{indent}   PV: {pv_str}")
        
        print()
        
        # Children
        if depth < max_depth:
            children_to_show = node.children[:max_children]
            for child in children_to_show:
                self._print_node(child, depth + 1, max_depth, max_children)
            
            if len(node.children) > max_children:
                print(f"{indent}   ... and {len(node.children) - max_children} more children")
                print()

class NoOpVisualizer:
    """No-operation visualizer that provides the same interface as SearchTreeVisualizer but does nothing."""
    
    def should_visualize(self, board: chess.Board) -> bool:
        """Always returns False - no visualization."""
        return False
    
    def start_search(self, board: chess.Board, depth: int):
        """No-op - does nothing."""
        pass
    
    def enter_node(self, board: chess.Board, move: Optional[chess.Move], depth: int, 
                   alpha: float, beta: float, quiescence: bool = False):
        """No-op - does nothing."""
        pass
    
    def exit_node(self, evaluation: float, node_type: str, principal_variation: List[str], 
                  nodes_searched: int, time_taken: float, tt_hit: bool = False, 
                  tt_depth: int = 0, evaluation_components: Optional[Dict[str, float]] = None,
                  move_order: Optional[List[str]] = None, beta_cutoff: bool = False):
        """No-op - does nothing."""
        pass
    
    def record_beta_cutoff(self, cutoff_move: str):
        """No-op - does nothing."""
        pass
    
    def finish_search(self, best_move: str, best_evaluation: float, total_nodes: int, 
                     total_time: float, final_pv: List[str]):
        """No-op - does nothing."""
        pass
    
    def export_tree_to_file(self, move_number: int = 1) -> str:
        """Returns empty string - no file export."""
        return ""
    
    def record_move_considered(self, move: chess.Move, board: chess.Board):
        """No-op - does nothing."""
        pass

# Global visualizer instance
_global_visualizer = SearchTreeVisualizer()

def get_visualizer() -> SearchTreeVisualizer:
    """Get the global search tree visualizer instance."""
    return _global_visualizer

def get_noop_visualizer() -> NoOpVisualizer:
    """Get a no-operation visualizer instance."""
    return NoOpVisualizer()

def configure_visualizer(enabled: bool = False, target_fen: Optional[str] = None):
    """Configure the global search tree visualizer."""
    global _global_visualizer
    _global_visualizer = SearchTreeVisualizer(enabled, target_fen)
