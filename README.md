# Python Chess Game

A graphical chess application built with Python, pygame, and python-chess library.

## Features

- **Graphical Chess Board**: Beautiful chessboard with high-quality piece images (60x60 native resolution)
- **Multiple Game Modes**:
  - Human vs Human
  - Human vs Computer (you play as White) - **Default Mode**
  - Computer vs Human (you play as Black)
  - **Checkmate Defense Mode** - Human (King only) vs Computer (King + Queen)
  - **Computer vs Computer** - Two engines play each other with speed tracking
- **Fixed Search Depth**: Engine uses consistent 4-ply search for fast, reliable play
- **Search Speed Tracking**: Monitor nodes per second and total search statistics
- **Move Validation**: All chess rules are enforced
- **Game Over Detection**: Automatically detects checkmate, stalemate, and draw
- **Smart Computer Player**: Uses minimax search with alpha-beta pruning and quiescence search
- **Crisp Graphics**: Uses native 60x60 pixel images for sharp, clear pieces
- **🎯 Endgame Tablebases**: Optional Syzygy tablebases for perfect endgame play

## Installation

1. **Install Python dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Optional: Install endgame tablebases** (recommended for better endgame play):
   ```bash
   # Download 3-4-5 piece tablebases (~1GB)
   # See "Endgame Tablebases" section below for detailed instructions
   ```

3. **Run the application**:
   ```bash
   python3 main_pygame.py
   ```

## Endgame Tablebases

The chess engine can use **Syzygy tablebases** for perfect endgame play. These provide optimal moves for positions with 7 or fewer pieces.

### Benefits
- **Perfect Endgame Play**: Optimal moves in complex endgames
- **Faster Play**: No need to search deep in endgame positions
- **Better Evaluation**: Accurate win/draw/loss assessment

### Manual Setup

**Step 1: Download Tablebases**
- Download 3-4-5 piece tablebases from: https://tablebase.sesse.net/syzygy/3-4-5/
- File size: ~1GB (3-5 piece endgames)

**Step 2: Create Directory Structure**
```bash
mkdir -p syzygy/3-4-5
```

**Step 3: Extract Files**
- Extract the downloaded files to the `syzygy/3-4-5/` directory
- Expected structure: `./syzygy/3-4-5/KPK.rtbw`, `./syzygy/3-4-5/KPK.rtbz`, etc.

**Step 4: Verify Installation**
The engine will show on startup:
```
✅ Endgame tablebases found and loaded from: ./syzygy/3-4-5
```

If not found:
```
⚠️  Syzygy tablebase not found. Endgame play will use standard evaluation.
💡 To improve endgame play, see README.md for download instructions.
   📁 Expected structure: ./syzygy/3-4-5/KPK.rtbw, ./syzygy/3-4-5/KPK.rtbz, etc.
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
- **5**: Computer vs Computer
- **ESC**: Quit

### Game Modes

- **Human vs Human**: Play both sides yourself
- **Human vs Computer**: You play as White, computer plays as Black (default)
- **Computer vs Human**: You play as Black, computer plays as White
- **Checkmate Defense Mode**: You play as White with only a King vs Computer with King + Queen
- **Computer vs Computer**: Watch two engines play each other with performance tracking

## Engine Features

### Computer Player

The computer player uses advanced chess engine techniques:

- **Minimax Search**: Looks ahead 4 moves consistently
- **Alpha-Beta Pruning**: Optimizes search by skipping obviously bad moves
- **Quiescence Search**: Continues searching captures to avoid horizon effect
- **Fixed Depth**: Consistent 4-ply search for reliable performance
- **Position Evaluation**: Considers material, piece-square tables, and tactical factors
- **Speed Tracking**: Monitors nodes per second for performance analysis
- **🎯 Endgame Tablebases**: Perfect play in endgames with 7 or fewer pieces

### Search Output

When the computer is thinking, you'll see:

```
🤔 Engine thinking (depth 4)...
🎭 Current side to move: Black
⏱️ Search completed in 0.59s
🏆 Best: Nf6 (1.5) | PV: Nf6 e5 Nd5 Nf3 | Speed: 20925 nodes/s
```

**With tablebases enabled:**
```
📊 Tablebase: WDL=1, DTZ=3
🎯 Using tablebase move: Qe8#
```

### Computer vs Computer Statistics

When watching computer vs computer games, you'll see:

```
📊 Game Stats: 15 moves, 187500 nodes, 12500 avg nodes/s
```

And at the end of the game:

```
🏁 FINAL GAME STATISTICS:
   Total moves: 45
   Total nodes searched: 562,500
   Total time: 45.23 seconds
   Average speed: 12,440 nodes/second
   Average nodes per move: 12,500
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
│   ├── engine.py          # Chess engine (minimax, evaluation, tablebases)
│   ├── player.py          # Player classes
│   └── pygame_gui.py      # Pygame GUI implementation
├── syzygy/
│   └── 3-4-5/            # Endgame tablebases (optional)
│       ├── KPK.rtbw
│       ├── KPK.rtbz
│       └── ... (other tablebase files)
└── assets/
    └── pieces/            # Chess piece images
```

## Technical Details

### Search Algorithm
- **Minimax**: Recursive search with alternating min/max levels
- **Alpha-Beta Pruning**: Reduces search space by pruning branches
- **Quiescence Search**: Continues searching captures at leaf nodes
- **Move Ordering**: Prioritizes captures and tactical moves first
- **Node Counting**: Tracks every position evaluated for speed analysis
- **🎯 Tablebase Integration**: Perfect play in endgame positions

### Evaluation Function
- **Material**: Primary factor (pawn=100, knight=320, bishop=330, rook=500, queen=900, king=20000)
- **Position**: Piece-square tables for positional bonuses (10% weight)
- **Tactics**: Checkmate detection, three-fold repetition, fifty-move rule
- **🎯 Tablebase Lookup**: Perfect evaluation for 7-piece or fewer endgames

### Performance
- **Fixed Depth**: Consistent 4-ply search for reliable speed
- **Efficient Search**: Captures prioritized, alpha-beta pruning
- **Real-time Feedback**: Shows search progress and best lines
- **Speed Tracking**: Nodes per second monitoring for performance analysis
- **🎯 Endgame Optimization**: Tablebase lookup for perfect endgame play

## Troubleshooting

### Common Issues

**Game won't start:**
- Ensure pygame is installed: `pip3 install pygame`
- Check that chess piece images are in `assets/pieces/`

**Tablebases not working:**
- Ensure tablebase files are in `./syzygy/3-4-5/` directory
- Check that files have .rtbw and .rtbz extensions
- Verify file permissions and disk space

**Poor endgame play:**
- Install tablebases by downloading from: https://tablebase.sesse.net/syzygy/3-4-5/
- Extract to `./syzygy/3-4-5/` folder in the project directory

### Tablebase Status

The engine will show tablebase status on startup:

```
✅ Syzygy tablebase loaded from: ./syzygy
```

Or if not available:

```
⚠️  Syzygy tablebase not found. Endgame play will use standard evaluation.
💡 To improve endgame play, see README.md for download instructions.
   📁 Expected structure: ./syzygy/3-4-5/KPK.rtbw, ./syzygy/3-4-5/KPK.rtbz, etc.
``` 