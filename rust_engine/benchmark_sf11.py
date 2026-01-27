"""
Stockfish 11 Benchmark (Rated by Stockfish 17)
Runs SF11 on the test suite and validates moves using SF17 to generate a Rating.
"""

import sys
import os
import subprocess
import re
import time
import argparse

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
    if sys.platform == 'win32':
        os.system('color')
    return f"{color}{text}{Colors.ENDC}"

# PATHS
SF11_PATH = r"c:\Users\vthie\.VScode\Project App\chess\engines\stockfish_20011801_x64_modern.exe"
SF17_PATH = r"c:\Users\vthie\.VScode\Project App\chess\engines\stockfish-windows-x86-64-avx2.exe"

# TEST POSITIONS
TEST_POSITIONS = [
    ("kasparov_topalov", "1rr3k1/4ppbp/2n3p1/2P5/p1BP4/P3P1P1/1B3P1P/3R1RK1 w - - 0 1"),
    ("tactical", "r2qk2r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 4 6"),
    ("endgame", "8/5pk1/6p1/8/5P2/6P1/5K2/8 w - - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("wac_001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"),
    ("italian_sharp", "r1bqk2r/pppp1ppp/2n2n2/2b1p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 5"),
    ("ruy_middle", "r1bqk2r/1ppp1ppp/p1n2n2/4p3/BbP1P3/5N2/PP1P1PPP/RNBQK2R w KQkq - 0 6"),
    ("qgd_complex", "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("tactical_pins", "r2qkb1r/1p1n1ppp/p2pbn2/4p3/4P3/1NN1B3/PPP1BPPP/R2QK2R w KQkq - 2 9"),
    ("endgame_rook", "8/5pk1/6p1/8/5P2/6P1/5K2/8 w - - 0 1"),
    ("isolated_qp", "r1bqkb1r/pp3ppp/2n1pn2/3pP3/3P4/P1N2N2/1P3PPP/R1BQKB1R w KQkq - 0 8"),
    ("discovered", "r1b1k2r/ppppqppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 6"),
    ("rook_endgame", "8/8/4k3/8/2p5/2P2K2/8/8 w - - 0 1"),
    ("double_edged", "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R1B2RK1 w - - 0 9"),
    ("king_exposed", "r1bq1rk1/pppp1ppp/5n2/2b5/2B1P3/5Q2/PPPP1PPP/RNB1K2R w KQ - 6 6"),
    ("knight_endgame", "8/4k3/8/2N5/2p5/2P2K2/8/8 w - - 0 1"),
]

class StockfishRunner:
    def __init__(self, binary_path):
        self.process = None
        self.path = binary_path

    def start(self):
        if self.process: return
        if not os.path.exists(self.path):
            print(colorize(f"ERROR: Stockfish not found at {self.path}", Colors.RED))
            sys.exit(1)
            
        self.process = subprocess.Popen(
            [self.path],
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

    def stop(self):
        if self.process:
            try: self.process.terminate()
            except: pass
            self.process = None

    def _send(self, cmd):
        if self.process:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def run_search(self, fen, depth):
        if not self.process: self.start()
        
        self._send("isready")
        while True:
            line = self.process.stdout.readline()
            if line.strip() == "readyok": break
            
        self._send("ucinewgame") 
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        
        stats = {"depth": 0, "time": 0, "nodes": 0, "nps": 0}
        best_move = "?"
        
        while True:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()
            
            if line.startswith("bestmove"):
                best_move = line.split()[1]
                break
                
            if line.startswith("info") and "depth" in line and "nodes" in line and "pv" in line:
                d_m = re.search(r'depth (\d+)', line)
                if d_m: stats["depth"] = int(d_m.group(1))
                
                n_m = re.search(r'nodes (\d+)', line)
                if n_m: stats["nodes"] = int(n_m.group(1))
                
                t_m = re.search(r'time (\d+)', line)
                if t_m: stats["time"] = int(t_m.group(1))
                
                nps_m = re.search(r'nps (\d+)', line)
                if nps_m: stats["nps"] = int(nps_m.group(1))
                
        return best_move, stats

    def get_top_moves(self, fen: str, depth: int = 12, num_moves: int = 5) -> list:
        if not self.process: self.start()
        
        self._send("isready")
        while True:
            line = self.process.stdout.readline()
            if line.strip() == "readyok": break

        self._send(f"setoption name MultiPV value {num_moves}")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        moves_by_mpv = {}
        
        while True:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()
            if line.startswith("bestmove"): break
                
            if 'info' in line and ' depth ' in line and ' pv ' in line:
                d_m = re.search(r' depth (\d+)', line)
                if not d_m: continue
                depth_val = int(d_m.group(1))
                
                mpv_match = re.search(r' multipv (\d+)', line)
                mpv_num = int(mpv_match.group(1)) if mpv_match else 1
                
                pv_match = re.search(r' pv ([a-h][1-8][a-h][1-8][qrbn]?)', line)
                if pv_match:
                    move = pv_match.group(1)
                    if mpv_num not in moves_by_mpv or depth_val > moves_by_mpv[mpv_num][0]:
                        moves_by_mpv[mpv_num] = (depth_val, move)
        
        self._send("setoption name MultiPV value 1") # Reset
        
        moves = []
        for i in range(1, num_moves + 1):
            if i in moves_by_mpv:
                moves.append(moves_by_mpv[i][1])
        return moves[:num_moves]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("depth", type=int, nargs="?", default=16)
    args = parser.parse_args()
    
    print("\n" + colorize("="*60, Colors.BLUE))
    print(colorize(f"  STOCKFISH 11 BENCHMARK (Rated by SF17, Depth {args.depth})", Colors.HEADER + Colors.BOLD))
    print(colorize("="*60, Colors.BLUE))
    print(f"{'Position':<20} {'Time':<10} {'Nodes':<10} {'NPS':<10} {'Move':<8} {'Quality':<10}")
    print("-" * 75)
    
    sf11 = StockfishRunner(SF11_PATH)
    sf17 = StockfishRunner(SF17_PATH)
    
    sf11.start()
    sf17.start()
    
    total_time = 0
    quality_hits = 0
    quality_total = 0
    quality_score = 0
    max_quality_score = 0
    
    try:
        for name, fen in TEST_POSITIONS:
            # 1. Benchmark Run (SF11)
            best_move, stats = sf11.run_search(fen, args.depth)
            
            time_ms = stats["time"]
            nodes = stats["nodes"]
            nps = stats["nps"]
            total_time += time_ms
            
            # 2. Quality Check (SF17)
            # Compare SF11's move against SF17's top 5
            judge_moves = sf17.get_top_moves(fen, depth=14, num_moves=5) # Judges deeper?
            
            max_quality_score += 5
            if best_move in judge_moves:
                position = judge_moves.index(best_move) + 1
                points = 6 - position
                quality_score += points
                quality_hits += 1
                quality_mark = f"★{position}"
            else:
                quality_mark = "✗"
            
            quality_total += 1
            
            # Display
            nps_str = f"{nps/1000:.0f}k" if nps < 1000000 else f"{nps/1000000:.1f}M"
            nodes_str = f"{nodes/1000:.0f}k"
            
            time_color = Colors.GREEN if time_ms < 1000 else (Colors.YELLOW if time_ms < 5000 else Colors.RED)
            quality_display = colorize(quality_mark, Colors.GREEN if "★" in quality_mark else Colors.RED)
            
            print(f"  {name:<18} {colorize(f'{time_ms:6.0f}ms', time_color)} {nodes_str:<9} {colorize(f'{nps_str:<9}', Colors.CYAN)} {colorize(f'{best_move:<7}', Colors.BOLD)} {quality_display}")
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        sf11.stop()
        sf17.stop()
        
    print("-" * 75)
    print(f"  Total Time:   {total_time/1000:.2f}s")
    
    quality_pct = (quality_score / max_quality_score) * 100 if max_quality_score > 0 else 0
    score_color = Colors.GREEN if quality_pct > 80 else Colors.RED
    print(f"  Move Quality: {quality_hits}/{quality_total}")
    print(f"  Quality Score: {colorize(f'{quality_score}/{max_quality_score} ({quality_pct:.1f}%)', score_color)}")
    
    if quality_pct >= 90:
        rating = colorize("★★★★★ GRANDMASTER", Colors.GREEN + Colors.BOLD)
    elif quality_pct >= 80:
        rating = colorize("★★★★☆ MASTER", Colors.GREEN)
    elif quality_pct >= 70:
        rating = colorize("★★★☆☆ EXPERT", Colors.CYAN)
    elif quality_pct >= 60:
        rating = colorize("★★☆☆☆ ADVANCED", Colors.BLUE)
    else:
        rating = colorize("★☆☆☆☆ BEGINNER", Colors.RED)
        
    print(f"  Rating: {rating}")

if __name__ == "__main__":
    main()
