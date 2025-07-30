from chess_game.pygame_gui import PygameChessGUI
import os
import chess.syzygy

def check_tablebases():
    """Check if Syzygy tablebases are available"""
    print("🔍 Checking for endgame tablebases...")
    
    # Check local syzygy/3-4-5 directory
    tablebase_path = "./syzygy/3-4-5"
    
    if os.path.exists(tablebase_path):
        try:
            # Try to open tablebases to verify they work
            tablebase = chess.syzygy.open_tablebase(tablebase_path)
            print(f"✅ Endgame tablebases found and loaded from: {tablebase_path}")
            print("🎯 The computer will play perfect endgames with 7 or fewer pieces!")
            return True
        except Exception as e:
            print(f"⚠️  Found tablebase directory but failed to load: {tablebase_path}")
            print(f"   Error: {e}")
    else:
        print("⚠️  No endgame tablebases found.")
        print("💡 To improve endgame play, see README.md for download instructions.")
        print("   📁 Expected structure: ./syzygy/3-4-5/KPK.rtbw, ./syzygy/3-4-5/KPK.rtbz, etc.")
        print("")
        print("🎮 The game will run normally without tablebases.")
        print("   The computer will use standard evaluation for endgames.")
    
    return False

if __name__ == "__main__":
    print("♔ Python Chess Game")
    print("=" * 40)
    
    # Check for tablebases before starting the game
    tablebases_available = check_tablebases()
    
    print("")
    print("🚀 Starting chess application...")
    print("=" * 40)
    
    app = PygameChessGUI()
    app.run() 