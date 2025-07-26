# Python Chess Game

A graphical chess application built with Python, pygame, and python-chess library.

## Features

- **Graphical Chess Board**: Beautiful chessboard with high-quality piece images (60x60 native resolution)
- **Multiple Game Modes**:
  - Human vs Human
  - Human vs Computer (you play as White) - **Default Mode**
  - Computer vs Human (you play as Black)
- **Move Validation**: All chess rules are enforced
- **Game Over Detection**: Automatically detects checkmate, stalemate, and draw
- **Simple Computer Player**: Currently makes random legal moves (ready for future AI integration)
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
- **ESC**: Quit

### Game Modes

- **Human vs Human**: Play both sides yourself
- **Human vs Computer**: You play as White, computer plays as Black (default)
- **Computer vs Human**: Computer plays as White, you play as Black

### Computer Player
The current computer player makes random legal moves. The architecture is designed to easily integrate more sophisticated AI in the future (similar to AlphaGo training approaches).

## Project Structure

```
chess-game/
├── main_pygame.py        # Entry point (pygame version)
├── main.py              # Entry point (tkinter version - deprecated)
├── chess_game/          # Main package
│   ├── __init__.py
│   ├── board.py         # Chess board logic
│   ├── player.py        # Player classes
│   ├── engine.py        # Computer player interface
│   ├── gui.py          # Tkinter GUI (deprecated)
│   └── pygame_gui.py   # Pygame GUI (current)
├── assets/
│   └── pieces/         # Chess piece images (60x60 PNG)
└── requirements.txt    # Python dependencies
```

## Future Enhancements

The application is designed to be easily extensible:

1. **AI Integration**: The `Engine` class in `chess_game/engine.py` provides an interface for plugging in neural networks or other AI algorithms
2. **Training Framework**: The modular design allows for training computer players using reinforcement learning
3. **Advanced Features**: Could add move history, analysis tools, opening databases, etc.

## Dependencies

- `python-chess`: Chess logic and move validation
- `pygame`: GUI framework for graphics and input handling
- `Pillow`: Image processing for chess pieces

## Troubleshooting

- **"pygame not found"**: Run `pip3 install pygame`
- **"python-chess not found"**: Run `pip3 install python-chess`
- **"PIL not found"**: Run `pip3 install Pillow`
- **Images not loading**: Ensure chess piece images are in `assets/pieces/` directory
- **Window not appearing**: Make sure you're running `main_pygame.py` (not `main.py`)

## Technical Notes

- **Display**: 480x480 pixel window (8 squares × 60 pixels each)
- **Images**: Native 60x60 PNG files for optimal quality
- **Framework**: Pygame for reliable cross-platform graphics
- **Performance**: 60 FPS refresh rate for smooth gameplay 