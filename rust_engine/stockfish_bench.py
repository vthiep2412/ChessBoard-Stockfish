"""
Compare Stockfish performance at depth 12
"""

import subprocess
import time

STOCKFISH_PATH = r"c:\Users\vthie\.VScode\Project App\Chess\engines\stockfish_20011801_x64_modern.exe"

TEST_POSITIONS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("sicilian", "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    ("kasparov_topalov", "1rr3k1/4ppbp/2n3p1/2P5/p1BP4/P3P1P1/1B3P1P/3R1RK1 w - - 0 1"),
    ("tactical", "r2qk2r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 4 6"),
    ("endgame", "8/5pk1/6p1/8/5P2/6P1/5K2/8 w - - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
]

def test_stockfish_position(fen: str, depth: int = 12) -> float:
    """Test Stockfish on a single position, return time in ms"""
    proc = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    commands = f"position fen {fen}\ngo depth {depth}\n"
    
    start = time.perf_counter()
    stdout, _ = proc.communicate(commands, timeout=120)
    elapsed = time.perf_counter() - start
    
    # Extract best move from output
    for line in stdout.split('\n'):
        if line.startswith('bestmove'):
            move = line.split()[1]
            break
    else:
        move = "???"
    
    return elapsed * 1000, move

def main():
    print("\n" + "="*50)
    print("  STOCKFISH BENCHMARK (depth 12)")
    print("="*50 + "\n")
    
    total_time = 0
    
    for name, fen in TEST_POSITIONS:
        print(f"  Testing: {name}...", end="", flush=True)
        time_ms, move = test_stockfish_position(fen, 12)
        print(f" {time_ms:8.1f}ms  Move: {move}")
        total_time += time_ms
    
    print("-"*50)
    print(f"  Total: {total_time:.1f}ms")
    print(f"  Average: {total_time/len(TEST_POSITIONS):.1f}ms per position")
    print()

if __name__ == "__main__":
    main()
