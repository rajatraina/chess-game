# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python chess engine and GUI application. The main components are:

- **Chess GUI** (`main_pygame.py`): Pygame-based graphical chess board with multiple game modes
- **Chess Engine** (`chess_game/engine.py`): Minimax engine with alpha-beta pruning, quiescence search, opening book, and optional NNUE evaluation
- **UCI Interface** (`uci_interface.py`): UCI protocol adapter for external GUI/online play
- **NNUE Trainer** (`trainer/`): PyTorch training pipeline for neural network evaluation models  
  - End-to-end CLI (split data → convert to `.features` → train): see [`trainer/README.md`](trainer/README.md). Run those commands from the repo root; training data paths typically live under `trainer/localdata/`. On macOS the README prefixes long jobs with `caffeinate -i` to prevent sleep during long conversion or training.
- **Lichess Bot** (`lichess-bot/`): Vendored bridge for online play (requires API token)

### Running the application

```bash
# GUI mode (requires DISPLAY)
DISPLAY=:1 python3 main_pygame.py < /dev/null

# UCI interface (stdin/stdout protocol)
python3 uci_interface.py
```

**Gotcha**: The GUI reads terminal input via `select.select` on stdin. When launching from a non-interactive shell, redirect stdin from `/dev/null` to prevent immediate exit.

### Running tests

```bash
# Engine correctness test (mate-in-1, early exit) — uses fixed-depth search
python3 -m pytest tests/ -v
# Or directly:
python3 tests/test_must_find_moves.py

# Speed benchmark (iterative deepening on a single position)
python3 main_pygame.py --speed
# Or directly:
python3 tests/speed_test.py
```

**Note**: The `test_must_find_moves.py` tests use a 30-second time budget by default, which triggers iterative deepening. The engine's `_predict_optimal_starting_depth` has a pre-existing argument mismatch in `predict_time()`. For quick engine validation, use fixed-depth mode (no `time_budget` argument) instead.

### Linting

No linter is configured in the repository. You can use flake8:

```bash
flake8 --max-line-length=150 --exclude=lichess-bot,trainer/localdata chess_game/ main_pygame.py uci_interface.py tests/
```

### Key notes

- ALSA audio warnings in the terminal are harmless (no sound card in the VM).
- Syzygy endgame tablebases and NNUE training data are optional; the engine works without them.
- The opening book (`opening_book.txt`, ~29MB) loads automatically on engine init.
- `torch` is a heavy dependency (~2GB) required for the NNUE trainer; it is listed in `requirements.txt`.
