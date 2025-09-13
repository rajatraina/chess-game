#!/usr/bin/env python3
"""
Create opening book from PGN games with evaluation data.

Usage:
    zstdcat lichess-games.pgn.zst | python create_opening_book_from_games.py

This script reads PGN games from stdin and creates an eval-based opening book
by tracking move sequences and their evaluation statistics. Only games with
evaluation annotations (%eval) are included.
"""

import sys
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Configuration options
MIN_ELO_RATING = 100  # Minimum ELO rating for both players
MAX_MOVES_TO_TRACK = 14  # Maximum number of moves to track in opening book
MIN_GAMES_FOR_POSITION = 10  # Minimum number of games required for a position to be included (reduced for eval data)
MIN_TIME_CONTROL_SECONDS = 10  # Minimum base time in seconds (e.g., 180 = 3 minutes)
MIN_MOVE_FREQUENCY_RATIO = 0.01  # Minimum frequency ratio for moves (e.g., 0.1 = 10% of most common move)
OUTPUT_FILE = "lichess-opening-book-eval.txt"
CHECKPOINT_INTERVAL = 10000000  # Write checkpoint every N games (10M)
VERBOSE_LOGGING = False  # Set to True to enable debug logging

# Constants for tracking whose turn it is
WHITE = "white"
BLACK = "black"

def parse_pgn_header(header_lines: List[str]) -> Dict[str, str]:
    """Parse PGN header tags."""
    header = {}
    for line in header_lines:
        if line.startswith('[') and line.endswith(']'):
            # Extract tag and value
            content = line[1:-1]
            if ' ' in content:
                tag, value = content.split(' ', 1)
                # Remove quotes from value
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                header[tag] = value
    return header

def parse_moves_with_eval(move_text: str) -> List[Tuple[str, float]]:
    """Parse move text and extract individual moves with their evaluations."""
    # VERBOSE LOGGING - DELETE WHEN DONE DEBUGGING
    if VERBOSE_LOGGING:
        print(f"DEBUG parse_moves_with_eval INPUT: {move_text[:100]}...")
    
    # Remove game result at the end
    move_text = re.sub(r'\s+(1-0|0-1|1/2-1/2)\s*$', '', move_text)
    
    moves_with_eval = []
    
    # Pattern to match moves with eval annotations
    # This captures: "1. e4 { [%eval 0.18] [%clk 0:03:00] }"
    # or "1... e5 { [%eval 0.22] [%clk 0:03:00] }"
    pattern = r'(\d+)(\.\.\.|\.)\s+([^\s]+)\s*\{\s*\[%eval\s+([^\]]+)\]\s*\[%clk[^]]*\]\s*\}'
    
    for match in re.finditer(pattern, move_text):
        move_num = int(match.group(1))
        move_type = match.group(2)  # "." for white, "..." for black
        move = match.group(3)
        eval_str = match.group(4)
        
        if move and not move.endswith('...'):
            # Strip evaluation markings (?, !, !!, ?!, etc.)
            move = re.sub(r'[?!]+$', '', move)
            
            # Parse evaluation value
            try:
                # Handle eval values like "0.18", "-0.11", "#+1", "#-2", etc.
                if eval_str.startswith('#'):
                    # Mate in X moves - convert to large positive/negative number
                    if eval_str.startswith('#+'):
                        eval_value = 100.0  # Large positive for mate advantage
                    elif eval_str.startswith('#-'):
                        eval_value = -100.0  # Large negative for mate disadvantage
                    else:
                        eval_value = 0.0
                else:
                    eval_value = float(eval_str)
                
                moves_with_eval.append((move, eval_value))
            except ValueError:
                # Skip moves with invalid eval values
                continue
    
    # VERBOSE LOGGING - DELETE WHEN DONE DEBUGGING
    if VERBOSE_LOGGING:
        print(f"DEBUG parse_moves_with_eval OUTPUT: {moves_with_eval}")
    
    return moves_with_eval

def get_game_result(result: str) -> str:
    """Convert PGN result to WDL format."""
    if result == "1-0":
        return "W"  # White wins
    elif result == "0-1":
        return "L"  # Black wins (from white's perspective)
    elif result == "1/2-1/2":
        return "D"  # Draw
    else:
        return "D"  # Default to draw for unknown results

def parse_time_control(time_control: str) -> int:
    """Parse TimeControl field and return base time in seconds."""
    try:
        # TimeControl format is typically "120+5" where 120 is base time in seconds
        # and +5 is increment in seconds
        if '+' in time_control:
            base_time_str = time_control.split('+')[0]
        else:
            base_time_str = time_control
        
        return int(base_time_str)
    except (ValueError, IndexError):
        return 0

def should_include_game(header: Dict[str, str], rejection_reasons: Dict[str, int]) -> bool:
    """Check if game meets inclusion criteria and track rejection reasons."""
    try:
        white_elo = int(header.get('WhiteElo', '0'))
        black_elo = int(header.get('BlackElo', '0'))
        
        # Both players must meet minimum ELO requirement
        if white_elo < MIN_ELO_RATING or black_elo < MIN_ELO_RATING:
            rejection_reasons['MinElo'] += 1
            return False
            
        # Game must have a valid result
        #result = header.get('Result', '')
        #if result not in ['1-0', '0-1', '1/2-1/2']:
        #    rejection_reasons['InvalidResult'] += 1
        #    return False
        
        # Game must have normal termination or time forfeit
        #termination = header.get('Termination', '')
        #if termination not in ['Normal', 'Time forfeit']:
        #    rejection_reasons['Termination'] += 1
        #    return False
        
        # Game must meet minimum time control requirement
        #time_control = header.get('TimeControl', '')
        #base_time_seconds = parse_time_control(time_control)
        #if base_time_seconds < MIN_TIME_CONTROL_SECONDS:
       #     rejection_reasons['TimeControl'] += 1
       #     return False
            
        return True
    except (ValueError, TypeError):
        rejection_reasons['ParseError'] += 1
        return False

def create_move_sequence_pairs_with_eval(moves_with_eval: List[Tuple[str, float]], max_moves: int = MAX_MOVES_TO_TRACK) -> List[Tuple[str, str, str, float]]:
    """Create pairs of (sequence, next_move, turn, eval) for opening book tracking."""
    # VERBOSE LOGGING - DELETE WHEN DONE DEBUGGING
    if VERBOSE_LOGGING:
        print(f"DEBUG create_move_sequence_pairs_with_eval INPUT: moves_with_eval={moves_with_eval}, max_moves={max_moves}")
    
    pairs = []
    
    # Extract just the moves for sequence building
    moves = [move for move, _ in moves_with_eval]
    
    # Handle starting position -> first move (White to move)
    if len(moves_with_eval) >= 1:
        move, eval_value = moves_with_eval[0]
        pairs.append(("", move, WHITE, eval_value))
    
    # Handle sequence -> next move pairs
    if max_moves >= 1 and len(moves_with_eval) >= 1:
        if len(moves_with_eval) >= 2:
            move, eval_value = moves_with_eval[1]
            pairs.append((f"1. {moves[0]}", move, BLACK, eval_value))
    if max_moves >= 2 and len(moves_with_eval) >= 2:
        if len(moves_with_eval) >= 3:
            move, eval_value = moves_with_eval[2]
            pairs.append((f"1. {moves[0]} {moves[1]}", move, WHITE, eval_value))
    if max_moves >= 3 and len(moves_with_eval) >= 3:
        if len(moves_with_eval) >= 4:
            move, eval_value = moves_with_eval[3]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]}", move, BLACK, eval_value))
    if max_moves >= 4 and len(moves_with_eval) >= 4:
        if len(moves_with_eval) >= 5:
            move, eval_value = moves_with_eval[4]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]}", move, WHITE, eval_value))
    if max_moves >= 5 and len(moves_with_eval) >= 5:
        if len(moves_with_eval) >= 6:
            move, eval_value = moves_with_eval[5]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]}", move, BLACK, eval_value))
    if max_moves >= 6 and len(moves_with_eval) >= 6:
        if len(moves_with_eval) >= 7:
            move, eval_value = moves_with_eval[6]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]}", move, WHITE, eval_value))
    if max_moves >= 7 and len(moves_with_eval) >= 7:
        if len(moves_with_eval) >= 8:
            move, eval_value = moves_with_eval[7]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]}", move, BLACK, eval_value))
    if max_moves >= 8 and len(moves_with_eval) >= 8:
        if len(moves_with_eval) >= 9:
            move, eval_value = moves_with_eval[8]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]}", move, WHITE, eval_value))
    if max_moves >= 9 and len(moves_with_eval) >= 9:
        if len(moves_with_eval) >= 10:
            move, eval_value = moves_with_eval[9]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]}", move, BLACK, eval_value))
    if max_moves >= 10 and len(moves_with_eval) >= 10:
        if len(moves_with_eval) >= 11:
            move, eval_value = moves_with_eval[10]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]}", move, WHITE, eval_value))
    if max_moves >= 11 and len(moves_with_eval) >= 11:
        if len(moves_with_eval) >= 12:
            move, eval_value = moves_with_eval[11]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]}", move, BLACK, eval_value))
    if max_moves >= 12 and len(moves_with_eval) >= 12:
        if len(moves_with_eval) >= 13:
            move, eval_value = moves_with_eval[12]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]}", move, WHITE, eval_value))
    if max_moves >= 13 and len(moves_with_eval) >= 13:
        if len(moves_with_eval) >= 14:
            move, eval_value = moves_with_eval[13]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]} 7. {moves[12]}", move, BLACK, eval_value))
    if max_moves >= 14 and len(moves_with_eval) >= 14:
        if len(moves_with_eval) >= 15:
            move, eval_value = moves_with_eval[14]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]} 7. {moves[12]} {moves[13]}", move, WHITE, eval_value))
    if max_moves >= 15 and len(moves_with_eval) >= 15:
        if len(moves_with_eval) >= 16:
            move, eval_value = moves_with_eval[15]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]} 7. {moves[12]} {moves[13]} 8. {moves[14]}", move, BLACK, eval_value))
    if max_moves >= 16 and len(moves_with_eval) >= 16:
        if len(moves_with_eval) >= 17:
            move, eval_value = moves_with_eval[16]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]} 7. {moves[12]} {moves[13]} 8. {moves[14]} {moves[15]}", move, WHITE, eval_value))
    if max_moves >= 17 and len(moves_with_eval) >= 17:
        if len(moves_with_eval) >= 18:
            move, eval_value = moves_with_eval[17]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]} 7. {moves[12]} {moves[13]} 8. {moves[14]} {moves[15]} 9. {moves[16]}", move, BLACK, eval_value))
    if max_moves >= 18 and len(moves_with_eval) >= 18:
        if len(moves_with_eval) >= 19:
            move, eval_value = moves_with_eval[18]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]} 7. {moves[12]} {moves[13]} 8. {moves[14]} {moves[15]} 9. {moves[16]} {moves[17]}", move, WHITE, eval_value))
    if max_moves >= 19 and len(moves_with_eval) >= 19:
        if len(moves_with_eval) >= 20:
            move, eval_value = moves_with_eval[19]
            pairs.append((f"1. {moves[0]} {moves[1]} 2. {moves[2]} {moves[3]} 3. {moves[4]} {moves[5]} 4. {moves[6]} {moves[7]} 5. {moves[8]} {moves[9]} 6. {moves[10]} {moves[11]} 7. {moves[12]} {moves[13]} 8. {moves[14]} {moves[15]} 9. {moves[16]} {moves[17]} 10. {moves[18]}", move, BLACK, eval_value))
    
    # VERBOSE LOGGING - DELETE WHEN DONE DEBUGGING
    if VERBOSE_LOGGING:
        print(f"DEBUG create_move_sequence_pairs_with_eval OUTPUT: {pairs}")
    
    return pairs

def calculate_eval_stats(eval_data: Dict[str, float]) -> Tuple[int, float]:
    """Calculate eval statistics: count and average eval."""
    count = eval_data.get('count', 0)
    total_eval = eval_data.get('total_eval', 0.0)
    
    if count == 0:
        return 0, 0.0
    
    avg_eval = total_eval / count
    return count, avg_eval

def write_opening_book(opening_book: Dict, output_file: str) -> None:
    """Write opening book to file with eval-based filtering applied."""
    with open(output_file, 'w') as f:
        # Iterate through each sequence in the opening book
        for sequence_str in sorted(opening_book.keys()):
            # Get all next moves for this sequence
            next_moves = opening_book[sequence_str]
            
            # Filter next moves by minimum game count
            filtered_moves = {}
            for next_move, eval_data in next_moves.items():
                count = eval_data.get('count', 0)
                if count >= MIN_GAMES_FOR_POSITION:
                    filtered_moves[next_move] = eval_data
            
            # Apply frequency ratio filtering
            if filtered_moves:
                # Find the most common move's game count
                most_common_games = max(eval_data.get('count', 0) for eval_data in filtered_moves.values())
                min_games_for_frequency = most_common_games * MIN_MOVE_FREQUENCY_RATIO
                
                # Filter out moves that are too rare compared to the most common move
                frequency_filtered_moves = {}
                for next_move, eval_data in filtered_moves.items():
                    count = eval_data.get('count', 0)
                    if count >= min_games_for_frequency:
                        frequency_filtered_moves[next_move] = eval_data
                
                filtered_moves = frequency_filtered_moves
            
            # Only write this sequence if it has moves with enough games
            if filtered_moves:
                f.write(f"MOVES: {sequence_str}\n")
                # VERBOSE LOGGING - DELETE WHEN DONE DEBUGGING
                if VERBOSE_LOGGING:
                    print(f"DEBUG: Writing MOVES: {sequence_str}")
                
                # Sort moves by average eval (descending for White, ascending for Black)
                def sort_key(item):
                    next_move, eval_data = item
                    count, avg_eval = calculate_eval_stats(eval_data)
                    current_turn = eval_data.get('turn', WHITE)  # Default to White
                    
                    if current_turn == WHITE:
                        # For White: higher eval is better (descending order)
                        return avg_eval
                    else:  # BLACK
                        # For Black: lower eval is better (ascending order)
                        # Use negative to reverse the sort order
                        return -avg_eval
                
                sorted_moves = sorted(
                    filtered_moves.items(),
                    key=sort_key,
                    reverse=True
                )
                
                # Write each next move with its eval stats
                for next_move, eval_data in sorted_moves:
                    count, avg_eval = calculate_eval_stats(eval_data)
                    f.write(f"{next_move} {count} | {avg_eval:.3f}\n")
                    # VERBOSE LOGGING - DELETE WHEN DONE DEBUGGING
                    if VERBOSE_LOGGING:
                        print(f"DEBUG: Wrote move {next_move} {count} | {avg_eval:.3f}")
                
                f.write("\n")

def main():
    """Main function to process PGN games and create opening book."""
    # Check if stdin is available
    if sys.stdin.isatty():
        print("Error: No input provided via stdin.")
        print("Usage: zstdcat lichess-games.pgn.zst | python3 create_opening_book_from_games.py")
        sys.exit(1)
    
    # Dictionary to store move sequences and their eval statistics
    # Key: move sequence string, Value: dict with eval data
    # Track opening book statistics: opening_book[sequence_str][next_move]['count'] and 'total_eval'
    # Also track whose turn it is: opening_book[sequence_str][next_move]['turn']
    opening_book = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'total_eval': 0.0, 'turn': WHITE}))
    
    games_processed = 0
    games_included = 0
    start_time = time.time()
    rejection_reasons = {
        'MinElo': 0,
        'InvalidResult': 0,
        'Termination': 0,
        'TimeControl': 0,
        'ParseError': 0,
        'NoEvalData': 0
    }
    
    print(f"Processing games with minimum ELO rating: {MIN_ELO_RATING}")
    print(f"Tracking up to {MAX_MOVES_TO_TRACK} moves per game")
    print(f"Minimum games per position: {MIN_GAMES_FOR_POSITION}")
    print("Only including games with evaluation data (%eval annotations)")
    print()
    
    # Stream games line by line from stdin
    current_game_lines = []
    in_game = False
    
    for line in sys.stdin:
        line = line.strip()
        
        if line.startswith('[Event '):
            # Start of new game
            if current_game_lines:
                # Process previous game
                games_processed += 1
                
                # Split game into headers and moves
                header_lines = []
                move_text = ""
                
                for game_line in current_game_lines:
                    if game_line.startswith('['):
                        header_lines.append(game_line)
                    else:
                        move_text += " " + game_line
                
                # Parse header
                header = parse_pgn_header(header_lines)
                
                # Check if game should be included
                if should_include_game(header, rejection_reasons):
                    # Parse moves with eval data
                    moves_with_eval = parse_moves_with_eval(move_text)
                    if moves_with_eval:
                        games_included += 1
                        
                        # Create move sequence pairs with eval data
                        sequence_pairs = create_move_sequence_pairs_with_eval(moves_with_eval)
                        
                        # Update statistics for each (sequence, next_move, turn, eval) pair
                        for sequence, next_move, turn, eval_value in sequence_pairs:
                            opening_book[sequence][next_move]['count'] += 1
                            opening_book[sequence][next_move]['total_eval'] += eval_value
                            # Store turn information (overwrites each time, but should be consistent)
                            opening_book[sequence][next_move]['turn'] = turn
                    else:
                        # Game has no eval data
                        rejection_reasons['NoEvalData'] += 1
                
                # Progress indicator and checkpoint
                if games_processed % 100000 == 0:
                    # Calculate processing rate
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        games_per_sec = games_processed / elapsed_time
                        if games_per_sec >= 1000:
                            rate_str = f"{games_per_sec/1000:.1f}K/sec"
                        else:
                            rate_str = f"{games_per_sec:.0f}/sec"
                    else:
                        rate_str = "0/sec"
                    
                    # Format large numbers with M suffix
                    def format_number(num):
                        if num >= 1000000:
                            return f"{num/1000000:.1f}M"
                        elif num >= 1000:
                            return f"{num/1000:.1f}K"
                        else:
                            return str(num)
                    
                    processed_str = format_number(games_processed)
                    included_str = format_number(games_included)
                    
                    rejected = games_processed - games_included
                    if rejected > 0:
                        rejection_pct = {}
                        for reason, count in rejection_reasons.items():
                            if count > 0:
                                pct = (count / games_processed) * 100.0
                                if pct >= 0.1:  # Only show if >= 0.1%
                                    rejection_pct[reason] = pct
                        
                        if rejection_pct:
                            rejection_str = ", ".join([f"{reason}: {pct:.1f}%" for reason, pct in rejection_pct.items()])
                            print(f"\rProcessed {processed_str} games ({rate_str}), included {included_str} (rejected: {rejection_str})", end="", flush=True)
                        else:
                            print(f"\rProcessed {processed_str} games ({rate_str}), included {included_str}", end="", flush=True)
                    else:
                        print(f"\rProcessed {processed_str} games ({rate_str}), included {included_str}", end="", flush=True)
                
                # Write checkpoint every CHECKPOINT_INTERVAL games
                if games_processed % CHECKPOINT_INTERVAL == 0 and games_processed > 0:
                    checkpoint_file = f"lichess-opening-book-eval.{games_processed}.txt"
                    print(f"\nWriting checkpoint: {checkpoint_file}")
                    write_opening_book(opening_book, checkpoint_file)
                    print(f"Checkpoint written: {checkpoint_file}")
            
            current_game_lines = [line]
            in_game = True
            
        elif in_game and line:
            current_game_lines.append(line)
    
    # Note: Final game is ignored as it may be incomplete
    
    # Format large numbers with M suffix
    def format_number(num):
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.1f}K"
        else:
            return str(num)
    
    print(f"\nProcessing complete!")
    print(f"Total games processed: {format_number(games_processed)}")
    print(f"Games included: {format_number(games_included)}")
    print(f"Unique sequences found: {format_number(len(opening_book))}")
    
    # Write final opening book to file
    print(f"\nWriting final opening book: {OUTPUT_FILE}")
    write_opening_book(opening_book, OUTPUT_FILE)
    
    print(f"Opening book written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()