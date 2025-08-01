# 🏆 Online Chess Play Guide

This guide will help you get your chess engine playing online and earning an ELO rating.

## 🚀 Quick Start Options

### **Option 1: Lichess Bot (Recommended)**

#### **Step 1: Create Lichess Bot Account**
1. Go to https://lichess.org/api#operation/botAccountCreate
2. Create a bot account
3. Get your API token

#### **Step 2: Install Lichess Bot Bridge**
```bash
pip install lichess-bot
```

#### **Step 3: Create Bot Configuration**
Create `lichess-bot.yml`:
```yaml
token: YOUR_API_TOKEN_HERE
engine: python3 uci_interface.py
```

#### **Step 4: Run the Bot**
```bash
python3 -m lichess-bot
```

### **Option 2: Chess.com Bot**

#### **Step 1: Apply for Bot Access**
1. Go to https://www.chess.com/club/chess-com-developer-community
2. Apply for bot access
3. Wait for approval

#### **Step 2: Use Chess.com API**
```bash
pip install chesscom
```

### **Option 3: FICS (Free Internet Chess Server)**

#### **Step 1: Install FICS Client**
```bash
pip install python-fics
```

#### **Step 2: Connect to FICS**
```python
import fics
client = fics.FICSClient()
client.connect()
```

## 🎯 **Testing Your Engine**

### **Local Testing with Arena**
1. Download Arena chess GUI: http://www.playwitharena.com/
2. Add your engine: `python3 uci_interface.py`
3. Test against other engines

### **Testing with Cutechess**
```bash
pip install cutechess-cli
cutechess-cli -engine cmd=python3 uci_interface.py -engine cmd=stockfish -each tc=10+0.1 proto=uci
```

## 📊 **Expected Performance**

Based on your engine's current strength:
- **Depth 4**: ~1200-1400 ELO
- **Depth 6**: ~1400-1600 ELO  
- **Depth 8**: ~1600-1800 ELO

## 🔧 **Optimization Tips**

### **For Better Performance:**
1. **Increase search depth** in `uci_interface.py`
2. **Add opening book** support
3. **Implement time management**
4. **Add endgame tablebase** support (already done!)

### **For Online Play:**
1. **Add move validation**
2. **Implement proper error handling**
3. **Add logging** for debugging
4. **Handle network timeouts**

## 🏆 **Getting an ELO Rating**

### **Lichess Bot Rating:**
- Play 10+ games to get provisional rating
- 30+ games for established rating
- Bot rating is separate from human rating

### **Chess.com Bot Rating:**
- Similar process to Lichess
- May require more games for established rating

## 🛠️ **Troubleshooting**

### **Common Issues:**
1. **Engine not responding**: Check UCI protocol implementation
2. **Timeouts**: Adjust thinking time settings
3. **Illegal moves**: Add move validation
4. **Connection issues**: Check network and API tokens

### **Debug Mode:**
Run with verbose output:
```bash
python3 uci_interface.py 2>&1 | tee engine.log
```

## 📈 **Next Steps**

1. **Start with Lichess** - easiest to set up
2. **Play some games** to test your engine
3. **Monitor performance** and adjust parameters
4. **Consider joining tournaments** for more games
5. **Optimize based on results**

## 🎮 **Alternative: Local Tournaments**

If online play seems complex, you can:
1. **Download other engines** (Stockfish, Leela, etc.)
2. **Run local tournaments** with Cutechess
3. **Compare performance** against known engines
4. **Get approximate ELO** from engine ratings

---

**Good luck with your chess engine! 🏆** 