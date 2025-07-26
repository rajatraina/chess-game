from chess_game.pygame_gui import PygameChessGUI

if __name__ == "__main__":
    print("Starting Pygame Chess...")
    print("Controls:")
    print("  Mouse: Click to select and move pieces")
    print("  N: New game")
    print("  1: Human vs Human")
    print("  2: Human vs Computer")
    print("  3: Computer vs Human")
    print("  ESC: Quit")
    
    app = PygameChessGUI()
    app.run() 