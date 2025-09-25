#!/usr/bin/env python3
"""
Model probing system for understanding chess evaluation models.

This module provides tools to probe neural network chess evaluation models
by comparing their outputs on specific position pairs that test different
chess concepts (material, position, tactics, etc.).
"""

import torch
import chess
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml

try:
    from transformer_model import create_model
    from data_loader import ChessDataLoader
except ImportError:
    # Handle case when running from trainer directory
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from trainer.transformer_model import create_model
    from trainer.data_loader import ChessDataLoader


class ModelProbe:
    """A probe that compares model evaluation between two FEN positions."""
    
    def __init__(self, fen1: str, fen2: str, description: str):
        """
        Initialize a model probe.
        
        Args:
            fen1: First FEN position
            fen2: Second FEN position  
            description: Human-readable description of what this probe tests
        """
        self.fen1 = fen1
        self.fen2 = fen2
        self.description = description
        
    def evaluate(self, model, feature_extractor, device, cp_to_prob_scale: float = 400.0) -> Dict[str, Any]:
        """
        Evaluate both positions with the model and return comparison results.
        
        Args:
            model: The neural network model
            feature_extractor: Feature extractor for converting positions to features
            device: Device to run inference on
            cp_to_prob_scale: Scale factor for centipawn to probability conversion
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'description': self.description,
            'fen1': self.fen1,
            'fen2': self.fen2,
            'fen1_eval': None,
            'fen2_eval': None,
            'eval_difference': None,
            'fen1_win_prob': None,
            'fen2_win_prob': None
        }
        
        try:
            # Convert FENs to boards
            board1 = chess.Board(self.fen1)
            board2 = chess.Board(self.fen2)
            
            # Extract features
            features1 = feature_extractor._board_to_features(board1)
            features2 = feature_extractor._board_to_features(board2)
            
            # Transpose to match expected format (18, 8, 8)
            features1 = features1.transpose(2, 0, 1)  # (8, 8, 18) -> (18, 8, 8)
            features2 = features2.transpose(2, 0, 1)  # (8, 8, 18) -> (18, 8, 8)
            
            # Convert to tensors and add batch dimension
            features1_tensor = torch.from_numpy(features1).float().unsqueeze(0).to(device)
            features2_tensor = torch.from_numpy(features2).float().unsqueeze(0).to(device)
            
            # Get model predictions
            model.eval()
            with torch.no_grad():
                win_prob1 = model(features1_tensor).cpu().item()
                win_prob2 = model(features2_tensor).cpu().item()
            
            # Convert win probabilities to centipawn evaluations
            def win_prob_to_cp(win_prob: float) -> float:
                import numpy as np
                win_prob_clamped = np.clip(win_prob, 0.001, 0.999)
                return -cp_to_prob_scale * np.log((1.0 - win_prob_clamped) / win_prob_clamped)
            
            eval1 = win_prob_to_cp(win_prob1)
            eval2 = win_prob_to_cp(win_prob2)
            eval_diff = eval1 - eval2
            
            results.update({
                'fen1_eval': eval1,
                'fen2_eval': eval2,
                'eval_difference': eval_diff,
                'fen1_win_prob': win_prob1,
                'fen2_win_prob': win_prob2
            })
            
        except Exception as e:
            results['error'] = str(e)
            
        return results


class ModelProbeSuite:
    """A collection of model probes for comprehensive evaluation."""
    
    def __init__(self):
        """Initialize with a comprehensive set of chess evaluation probes."""
        self.probes = []
        self._add_material_probes()
        self._add_positional_probes()
        self._add_tactical_probes()
        self._add_endgame_probes()
    
    def _add_material_probes(self):
        """Add probes for material evaluation."""
        # Starting position vs missing white queen
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            "Queen material value"
        ))
        
        # Starting position vs missing white rook
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w KQkq - 0 1",
            "Rook material value"
        ))
        
        # Starting position vs missing white bishop
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1",
            "Bishop material value"
        ))
        
        # Starting position vs missing white knight
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "Knight material value"
        ))
        
        # Starting position vs missing white C pawn
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1",
            "C Pawn material value"
        ))

        # Starting position vs missing white D pawn
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
            "D Pawn material value"
        ))
    

    def _add_positional_probes(self):
        """Add probes for positional evaluation."""
        # Tempo
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "One tempo in start pos"
        ))

        # Center control
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "Center control (e4 pawn)"
        ))
        
        # Development
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 1",
            "Development (Nf3)"
        ))
        
        # King safety
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            "King safety (castling rights)"
        ))
    
    def _add_tactical_probes(self):
        """Add probes for tactical evaluation."""
        # Pin
        # Fork
        self.probes.append(ModelProbe(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkb1r/1ppp1ppp/p4n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 4",
            "Italian Nxf7 fork threat (+440cp)"
        ))
    
    def _add_endgame_probes(self):
        """Add probes for endgame evaluation."""
        # King and pawn vs king
        self.probes.append(ModelProbe(
            "8/8/8/8/8/4K3/8/4k3 w - - 0 1",
            "8/8/8/8/8/4K3/8/4k3 w - - 0 1",
            "King vs King (equal)"
        ))
        
        # King and pawn vs king (with pawn)
        self.probes.append(ModelProbe(
            "5k2/8/5K2/8/8/8/8/8 w - - 15 8",
            "5k2/8/5K2/5P2/8/8/8/8 b - - 15 8",
            "KP vs K (K opposition/winning)"
        ))
        
    
    def add_probe(self, probe: ModelProbe):
        """Add a custom probe to the suite."""
        self.probes.append(probe)
    
    def run_all_probes(self, model, feature_extractor, device, cp_to_prob_scale: float = 400.0) -> List[Dict[str, Any]]:
        """
        Run all probes and return results.
        
        Args:
            model: The neural network model
            feature_extractor: Feature extractor
            device: Device to run inference on
            cp_to_prob_scale: Scale factor for centipawn to probability conversion
            
        Returns:
            List of probe results
        """
        results = []
        for probe in self.probes:
            result = probe.evaluate(model, feature_extractor, device, cp_to_prob_scale)
            results.append(result)
        return results
    
    def print_results(self, results: List[Dict[str, Any]]):
        """Print probe results in a formatted table."""
        print(f"\n  🔍 Model Probe Results:")
        print(f"  {'Description':<30} {'FEN1 Eval':<10} {'FEN2 Eval':<10} {'Difference':<12} {'Status':<8}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
        
        for result in results:
            if 'error' in result:
                print(f"  {result['description']:<30} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'FAIL':<8}")
            else:
                eval1 = result['fen1_eval']
                eval2 = result['fen2_eval']
                diff = result['eval_difference']
                
                # Determine if the evaluation makes sense
                status = "✓" if abs(diff) > 10 else "?"  # Threshold for meaningful difference
                
                print(f"  {result['description']:<30} {eval1:<10.1f} {eval2:<10.1f} {diff:<12.1f} {status:<8}")


def run_probes_from_config(config_path: str, model_path: str = None):
    """
    Run model probes using configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        model_path: Path to model checkpoint (optional, uses best model if not provided)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    # Load model weights
    if model_path is None:
        model_path = "checkpoints/best_model.pth"
    
    if Path(model_path).exists():
        try:
            # Try loading with weights_only=True first (safer)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            # Silently fall back to weights_only=False (expected for our checkpoints)
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            except Exception as e2:
                print(f"Failed to load checkpoint: {e2}")
                print("Using random weights instead")
                checkpoint = None
        
        if checkpoint is not None:
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Full trainer checkpoint
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from trainer checkpoint {model_path}")
            else:
                # Direct model weights
                model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
        else:
            print("Using random weights")
    else:
        print(f"Model file {model_path} not found, using random weights")
    
    # Create feature extractor
    from chess_game.neural_network_evaluator import NeuralNetworkEvaluator
    feature_extractor = NeuralNetworkEvaluator()
    
    # Create probe suite
    probe_suite = ModelProbeSuite()
    
    # Run probes
    results = probe_suite.run_all_probes(
        model, 
        feature_extractor, 
        device, 
        config.get('cp_to_prob_scale', 400.0)
    )
    
    # Print results
    probe_suite.print_results(results)


def main():
    """Main function for standalone script usage."""
    parser = argparse.ArgumentParser(description='Run model probes to understand chess evaluation')
    parser.add_argument('--config', type=str, default='trainer/config_medium.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    
    args = parser.parse_args()
    
    run_probes_from_config(args.config, args.model)


if __name__ == "__main__":
    # Add current directory to path to handle imports when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
