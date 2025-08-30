#!/usr/bin/env python3

import unittest
import chess
import sys
import os

# Add the parent directory to the path to import chess_game
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chess_game.engine import MinimaxEngine

class TestMustFindMoves(unittest.TestCase):
    """Test that the search finds the correct moves in specific positions"""
    
    def setUp(self):
        """Set up the test environment"""
        self.engine = MinimaxEngine()
        # Use 30-second time budget for thorough testing
        self.time_budget = 30.0
    
    def test_expected_moves(self):
        """Test that the engine finds the expected best moves in specific positions"""
        
        # Test cases: (FEN, side_to_move, expected_best_move, description)
        test_cases = [
            (
                "6k1/6pp/3p3n/p2N4/1pqP1P2/5P1P/PPP2P2/4Q1K1 w - - 1 23",
                "white",
                "Qe8#",
                "White to move, should find checkmate Qe8#"
            ),
            (
                "8/8/8/8/8/8/8/K7 w - - 0 1",
                "white",
                "Kb2",
                "Lone king, should move to corner (tablebase choice)"
            ),
            # Add more test cases here as needed:
            # (
            #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #     "white", 
            #     "e4",
            #     "Starting position, White should play e4"
            # ),
            # (
            #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            #     "black",
            #     "Nf6",
            #     "After 1.e4 e5, Black should develop knight"
            # ),
        ]
        
        for i, (fen, side_to_move, expected_move, description) in enumerate(test_cases, 1):
            with self.subTest(f"Test case {i}: {description}"):
                print(f"\n{'='*60}")
                print(f"Test case {i}: {description}")
                print(f"FEN: {fen}")
                print(f"Side to move: {side_to_move}")
                print(f"Expected best move: {expected_move}")
                print(f"{'='*60}")
                
                # Set up the position
                board = chess.Board(fen)
                
                # Verify the side to move matches expectation
                expected_turn = chess.WHITE if side_to_move == "white" else chess.BLACK
                self.assertEqual(board.turn, expected_turn, 
                               f"Board turn ({board.turn}) doesn't match expected ({expected_turn})")
                
                # Get legal moves for reference
                legal_moves = [board.san(move) for move in board.legal_moves]
                print(f"Legal moves: {legal_moves}")
                
                # Verify expected move is legal
                self.assertIn(expected_move, legal_moves, 
                             f"Expected move '{expected_move}' is not legal in this position")
                
                # Get the engine's best move
                best_move = self.engine.get_move(board, time_budget=self.time_budget)
                best_move_san = board.san(best_move)
                
                print(f"Engine chose: {best_move_san}")
                print(f"Engine evaluation: {self.engine.best_value}")
                
                # Verify the engine chose the expected move
                self.assertEqual(best_move_san, expected_move,
                               f"Engine chose '{best_move_san}' but expected '{expected_move}'")
                
                print(f"✅ PASS: Engine correctly chose {best_move_san}")
    
    def test_early_exit_functionality(self):
        """Test that the engine can exit early with a very short time budget"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        
        print(f"\n{'='*60}")
        print("Testing early exit functionality")
        print(f"FEN: {fen}")
        print(f"Time budget: 0.1 seconds")
        print(f"{'='*60}")
        
        # Use a very short time budget to trigger early exit
        short_time_budget = 0.1
        
        # Get the engine's best move with short time budget
        best_move = self.engine.get_move(board, time_budget=short_time_budget)
        best_move_san = board.san(best_move)
        
        print(f"Engine chose: {best_move_san}")
        print(f"Engine evaluation: {self.engine.best_value}")
        
        # Verify the engine returned a valid move (even with early exit)
        self.assertIsNotNone(best_move, "Engine should return a valid move even with early exit")
        self.assertIn(best_move_san, [board.san(move) for move in board.legal_moves], 
                     f"Engine returned invalid move: {best_move_san}")
        
        print(f"✅ PASS: Engine correctly handled early exit and returned valid move: {best_move_san}")

def run_tests():
    """Run the tests and return results"""
    print("Running expected moves tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMustFindMoves)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
