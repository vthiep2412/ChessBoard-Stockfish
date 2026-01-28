import sys
import os
import subprocess
import benchmark
# Keep existing elements from benchmark
from benchmark import TEST_POSITIONS, Colors, colorize

class OwnStockfishHelper:
    """
    Internal Stockfish helper with specific path resolution:
    ../engines/stockfish-windows-x86-64-avx2.exe
    """
    def __init__(self):
        self.process = None

    def start(self):
        # 1. Get current script directory
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up to parent directory (..) -> Go into 'engines' -> Target specific exe
        engine_path = os.path.abspath(os.path.join(
            current_script_dir, 
            "..", 
            "engines", 
            "stockfish-windows-x86-64-avx2.exe"
        ))

        print(f"Looking for engine at: {engine_path}")

        if not os.path.exists(engine_path):
            print(colorize(f"Error: Engine not found at {engine_path}", Colors.RED))
            return

        try:
            self.process = subprocess.Popen(
                [engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            self._send("uci")
            
            # Wait for 'uciok' to confirm it's running
            if self._wait_for("uciok"):
                return
                
        except Exception as e:
            print(colorize(f"Error starting engine: {e}", Colors.RED))
            self.process = None

    def _send(self, cmd):
        if self.process:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def _wait_for(self, target_text):
        if not self.process:
            return False
        while True:
            line = self.process.stdout.readline()
            if not line: break
            if target_text in line:
                return True
        return False

    def get_top_moves(self, fen, depth=20, num_moves=5):
        if not self.process:
            return []

        # Reset state, set MultiPV
        self._send("stop")
        self._send("isready")
        self._wait_for("readyok")
        self._send(f"setoption name MultiPV value {num_moves}")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        collected_moves = {}

        while True:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()

            if line.startswith("bestmove"):
                break

            # Parse 'info' lines for MultiPV data
            if "depth" in line and "multipv" in line and "pv" in line:
                try:
                    parts = line.split()
                    
                    d_idx = parts.index("depth")
                    current_depth = int(parts[d_idx + 1])
                    
                    # Ensure we get moves at the requested depth
                    if current_depth == depth:
                        mpv_idx = parts.index("multipv")
                        rank = int(parts[mpv_idx + 1])
                        
                        pv_idx = parts.index("pv")
                        move = parts[pv_idx + 1]
                        
                        collected_moves[rank] = move
                except (ValueError, IndexError):
                    continue

        return [collected_moves[i] for i in sorted(collected_moves.keys())]

    def stop(self):
        if self.process:
            self._send("quit")
            try:
                self.process.terminate()
            except:
                pass
            self.process = None


def generate_cache():
    print(colorize("============================================", Colors.BLUE))
    print(colorize("  Pre-generating Stockfish Best Moves", Colors.BOLD))
    print(colorize("============================================", Colors.BLUE))
    
    sf = OwnStockfishHelper()
    sf.start()
    
    if not sf.process:
        print("Could not start Stockfish. Please check the path.")
        return

    cached_data = {}

    for name, fen in TEST_POSITIONS:
        print(f"Analyzing {name}...", end="", flush=True)
        # Depth 20 as requested, top 5 moves
        moves = sf.get_top_moves(fen, depth=20, num_moves=5)
        print(f" Done. Top moves: {moves}")
        cached_data[name] = moves
    
    sf.stop()
    
    print("\n" + colorize("Generating code block...", Colors.GREEN))
    print("Replace the 'TEST_POSITIONS' list in benchmark.py with this structure if you want to bundle it,")
    print("OR I can patch benchmark.py to use this cache dictionary.")
    
    print("\nCACHED_BEST_MOVES = {")
    for name, moves in cached_data.items():
        moves_str = ", ".join([f"'{m}'" for m in moves])
        print(f"    '{name}': [{moves_str}],")
    print("}")

if __name__ == "__main__":
    generate_cache()