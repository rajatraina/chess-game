# Python Chess Game

A graphical chess application built with Python, tkinter, and python-chess library.

## Features

- **Graphical Chess Board**: Beautiful chessboard with piece images
- **Multiple Game Modes**:
  - Human vs Human
  - Human vs Computer (you play as White)
  - Computer vs Human (you play as Black)
- **Move Validation**: All chess rules are enforced
- **Game Over Detection**: Automatically detects checkmate, stalemate, and draw
- **Simple Computer Player**: Currently makes random legal moves (ready for future AI integration)

## Installation

1. **Install Python dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python3 main.py
   ```

## How to Play

### Starting the Game
- The application will open in a new window
- Use the "Game" menu to select your preferred game mode
- Click "New Game" to start a fresh game

### Making Moves
1. **Click on a piece** to select it (it will be highlighted in yellow)
2. **Click on a destination square** to move the piece
3. If the move is legal, it will be executed
4. If the move is illegal, you can select a different piece

### Game Modes

- **Human vs Human**: Play both sides yourself
- **Human vs Computer**: You play as White, computer plays as Black
- **Computer vs Human**: Computer plays as White, you play as Black

### Computer Player
The current computer player makes random legal moves. The architecture is designed to easily integrate more sophisticated AI in the future (similar to AlphaGo training approaches).

## Project Structure

```
chess-game/
├── main.py              # Entry point
├── chess_game/          # Main package
│   ├── __init__.py
│   ├── board.py         # Chess board logic
│   ├── player.py        # Player classes
│   ├── engine.py        # Computer player interface
│   └── gui.py          # Graphical user interface
├── assets/
│   └── pieces/         # Chess piece images
└── requirements.txt    # Python dependencies
```

## Future Enhancements

The application is designed to be easily extensible:

1. **AI Integration**: The `Engine` class in `chess_game/engine.py` provides an interface for plugging in neural networks or other AI algorithms
2. **Training Framework**: The modular design allows for training computer players using reinforcement learning
3. **Advanced Features**: Could add move history, analysis tools, opening databases, etc.

## Dependencies

- `python-chess`: Chess logic and move validation
- `Pillow`: Image processing for chess pieces
- `tkinter`: GUI framework (included with Python)

## Troubleshooting

- **"python-chess not found"**: Run `pip3 install python-chess`
- **"PIL not found"**: Run `pip3 install Pillow`
- **Images not loading**: Ensure chess piece images are in `assets/pieces/` directory 