# Python Chess Game

A graphical chess application built with Python, pygame, and python-chess library.

## Features

- **Graphical Chess Board**: Beautiful chessboard with high-quality piece images (60x60 native resolution)
- **Multiple Game Modes**:
  - Human vs Human
  - Human vs Computer (you play as White) - **Default Mode**
  - Computer vs Human (you play as Black)
  - **Checkmate Defense Mode** - Human (King only) vs Computer (King + Queen)
- **Adaptive Search Depth**: Engine automatically adjusts search depth based on time (4-10 plies)
- **Move Validation**: All chess rules are enforced
- **Game Over Detection**: Automatically detects checkmate, stalemate, and draw
- **Smart Computer Player**: Uses minimax search with alpha-beta pruning and quiescence search
- **Crisp Graphics**: Uses native 60x60 pixel images for sharp, clear pieces

## Installation

1. **Install Python dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python3 main_pygame.py
   ```

## How to Play

### Starting the Game
- The application will open in a new window (480x480 pixels)
- The game starts in **Human vs Computer** mode by default
- You play as White, computer plays as Black

### Making Moves
1. **Click on a piece** to select it (it will be highlighted in yellow)
2. **Click on a destination square** to move the piece
3. If the move is legal, it will be executed
4. If the move is illegal, you can select a different piece

### Controls

**Mouse:**
- Left click to select and move pieces

**Keyboard:**
- **N**: New game
- **1**: Human vs Human
- **2**: Human vs Computer (you as White)
- **3**: Computer vs Human (you as Black)
- **4**: Checkmate Defense Mode
- **ESC**: Quit

### Game Modes

- **Human vs Human**: Play both sides yourself
- **Human vs Computer**: You play as White, computer plays as Black (default)
- **Computer vs Human**: Computer plays as White, you play as Black
- **Checkmate Defense Mode**: You play as White with only a King, computer plays as Black with King + Queen

### Checkmate Defense Mode

This special mode challenges you to survive as long as possible against a computer with a Queen advantage:

- **Your pieces**: White King only
- **Computer pieces**: Black King + Queen
- **Goal**: Avoid checkmate for as long as possible
- **Strategy**: Use your King to escape and avoid the Queen's attacks
- **Challenge**: The computer will try to checkmate you efficiently

### Computer Player

The computer player uses advanced chess engine techniques:

- **Minimax Search**: Looks ahead 4-10 moves depending on position complexity
- **Alpha-Beta Pruning**: Optimizes search by skipping obviously bad moves
- **Quiescence Search**: Continues searching captures to avoid horizon effect
- **Adaptive Depth**: Automatically adjusts search depth based on time:
  - Starts at 4 plies minimum
  - Increases depth if search completes under 10 seconds
  - Decreases depth if search takes longer than 10 seconds
  - Maximum depth of 10 plies
- **Position Evaluation**: Considers material, piece-square tables, and tactical factors

### Engine Output

The engine provides detailed analysis during play:

```
🤔 Engine thinking (depth 4)...
🎭 Current side to move: Black
  🎯 d7d6: -120 | PV: d7d6 e4e5 d6e5
  🌟 New best! d7d6: -120 | Variation: d7d6
⏱️ Search completed in 2.34s (under 10s limit)
📈 Increasing depth to 5 for next move
🏆 Best: d7d6 (-120) | PV: d7d6 e4e5 d6e5 | Depth: 5
```

## Project Structure

```
chess-game/
├── main_pygame.py          # Main entry point
├── requirements.txt         # Python dependencies
├── README.md              # This file
├── chess_game/
│   ├── __init__.py
│   ├── board.py           # Chess board wrapper
│   ├── engine.py          # Chess engine (minimax, evaluation)
│   ├── player.py          # Player classes
│   └── pygame_gui.py      # Pygame GUI implementation
└── assets/
    └── pieces/            # Chess piece images
```

## Technical Details

### Search Algorithm
- **Minimax**: Recursive search with alternating min/max levels
- **Alpha-Beta Pruning**: Reduces search space by pruning branches
- **Quiescence Search**: Continues searching captures at leaf nodes
- **Move Ordering**: Prioritizes captures and tactical moves first

### Evaluation Function
- **Material**: Primary factor (pawn=100, knight=320, bishop=330, rook=500, queen=900, king=20000)
- **Position**: Piece-square tables for positional bonuses (10% weight)
- **Tactics**: Checkmate detection, three-fold repetition, fifty-move rule

### Performance
- **Adaptive Depth**: 4-10 plies based on position complexity
- **Time Management**: 10-second time limit per move
- **Efficient Search**: Captures prioritized, alpha-beta pruning
- **Real-time Feedback**: Shows search progress and best lines

## Troubleshooting

### Common Issues

**Game won't start:**
- Ensure pygame is installed: `pip3 install pygame`
- Check that chess piece images are in `assets/pieces/`

**Engine seems slow:**
- The adaptive depth system will automatically adjust
- Complex positions may take longer to analyze
- You can see the current search depth in the output

**Checkmate Defense Mode too hard:**
- This is intentionally challenging!
- Focus on keeping your King away from the center
- Try to use the board edges and corners for safety

## Future Enhancements

- Opening book integration
- Endgame tablebases
- Network play
- Save/load games
- Analysis mode with move suggestions 