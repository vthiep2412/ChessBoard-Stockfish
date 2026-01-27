"""
Custom Chess Engine Benchmark
Allows inputting custom FEN and depth to test the engine.
"""

import time
import sys
import os
import subprocess
import re

# Add the target directory to path for the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target', 'release'))

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize(text, color):
    """Wrap text in color codes if supported"""
    if sys.platform == 'win32':
        os.system('color')  # Enable ANSI support in Windows console
    return f"{color}{text}{Colors.ENDC}"

# Stockfish path for move validation (optional)
STOCKFISH_PATH = r"c:\Users\vthie\.VScode\Project App\Chess\engines\stockfish-avx2.exe"

try:
    import rust_engine
    # CRITICAL: Clear poisoned TT entries from previous crashed/failed searches!
    rust_engine.clear_tt()
    print(colorize("  ✓ TT cleared - engine ready", Colors.GREEN))
except ImportError:
    print(colorize("ERROR: rust_engine module not found!", Colors.RED))
    print("Make sure to run: cargo build --release")
    print("Or run: maturin develop --release")
    sys.exit(1)

# Reuse StockfishHelper if needed, generic version
class StockfishHelper:
    def __init__(self):
        self.process = None

    def start(self):
        if self.process: return
        if not os.path.exists(STOCKFISH_PATH): return
        try:
            self.process = subprocess.Popen(
                [STOCKFISH_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self._send("uci")
            while True:
                line = self.process.stdout.readline()
                if not line or line.strip() == "uciok": break
        except Exception as e:
            print(f"SF Error: {e}")

    def stop(self):
        if self.process:
            try: self.process.terminate()
            except: pass
            self.process = None

    def _send(self, cmd):
        if self.process:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
            except IOError: self.stop()

    def get_evaluation(self, fen: str, depth: int = 12):
        """Get quick stockfish evaluation for comparison"""
        if not self.process: self.start()
        if not self.process: return None

        self._send("isready")
        while True:
            line = self.process.stdout.readline()
            if not line or line.strip() == "readyok": break

        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        
        best_move = "?"
        score = "?"
        
        while True:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()
            if line.startswith("bestmove"):
                best_move = line.split()[1]
                break
            if 'score cp' in line:
                m = re.search(r'score cp (-?\d+)', line)
                if m: score = f"{int(m.group(1))/100:.2f}"
            elif 'score mate' in line:
                m = re.search(r'score mate (-?\d+)', line)
                if m: score = f"M{m.group(1)}"
                
        return best_move, score

def main():
    print("\n" + colorize("="*60, Colors.BLUE))
    print(colorize("  CUSTOM RUST ENGINE BENCHMARK", Colors.HEADER + Colors.BOLD))
    print(colorize("  Enter 'exit' or press Ctrl+C to quit", Colors.YELLOW))
    print(colorize("="*60, Colors.BLUE))
    
    sf = StockfishHelper()
    
    while True:
        try:
            print(f"\n{colorize('Input FEN', Colors.BOLD)} (Press Enter for startpos):")
            fen = input("> ").strip()
            if fen.lower() in ['exit', 'quit', 'q']:
                break
            
            if not fen:
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                print(f"Using startpos")
            
            print(f"{colorize('Input Depth', Colors.BOLD)} (Default 10):")
            depth_str = input("> ").strip()
            if depth_str.lower() in ['exit', 'quit', 'q']:
                break
            
            depth = int(depth_str) if depth_str.isdigit() else 10
            
            print(f"\n{colorize('Running search...', Colors.CYAN)}")
            print("-" * 50)
            
            # Run Engine
            start = time.perf_counter()
            try:
                # Disable debug prints for speed
                try: rust_engine.set_debug(False)
                except: pass
                
                best_move = rust_engine.get_best_move(fen, depth, 5, False)
            except Exception as e:
                print(colorize(f"Error: {e}", Colors.RED))
                continue
                
            elapsed = time.perf_counter() - start
            
            # Stats
            nodes, qnodes = rust_engine.get_node_counts()
            total_nodes = nodes + qnodes
            nps = int(total_nodes / elapsed) if elapsed > 0.001 else 0
            nps_str = f"{nps/1000:.0f}k" if nps < 1000000 else f"{nps/1000000:.2f}M"
            
            # engine eval
            eng_eval = rust_engine.evaluate(fen)
            
            print(f"  {colorize('Best Move:', Colors.GREEN)} {best_move}")
            print(f"  {colorize('Time:', Colors.YELLOW)}      {elapsed*1000:.1f} ms")
            print(f"  {colorize('Nodes:', Colors.BLUE)}     {total_nodes:,} ({nps_str} NPS)")
            print(f"  {colorize('Static Eval:', Colors.CYAN)} {eng_eval}")
            
            # Stockfish Comparison (Optional)
            if os.path.exists(STOCKFISH_PATH):
                print(f"\n  {colorize('checking stockfish...', Colors.CYAN)}")
                sf_res = sf.get_evaluation(fen, 12)
                if sf_res:
                    sf_move, sf_score = sf_res
                    match = "MATCH ✓" if sf_move == best_move else "DIFF ✗"
                    color = Colors.GREEN if sf_move == best_move else Colors.RED
                    print(f"  Stockfish: {sf_move} ({sf_score})  [{colorize(match, color)}]")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            
    sf.stop()

if __name__ == "__main__":
    main()
