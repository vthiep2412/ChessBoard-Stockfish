
import sys
import os
import benchmark
from benchmark import StockfishHelper, TEST_POSITIONS, Colors, colorize

def generate_cache():
    print(colorize("============================================", Colors.BLUE))
    print(colorize("  Pre-generating Stockfish Best Moves", Colors.BOLD))
    print(colorize("============================================", Colors.BLUE))
    
    sf = StockfishHelper()
    sf.start()
    if not sf.process:
        print("Could not start Stockfish found in benchmark.py")
        return

    cached_data = {}

    for name, fen in TEST_POSITIONS:
        print(f"Analyzing {name}...", end="", flush=True)
        # Depth 20 as requested by user, top 5 moves
        moves = sf.get_top_moves(fen, depth=20, num_moves=5)
        print(f" Done. Top moves: {moves}")
        cached_data[name] = moves
    
    sf.stop()
    
    print("\n" + colorize("Generating code block...", Colors.GREEN))
    print("Replace the 'TEST_POSITIONS' list in benchmark.py with this structure if you want to bundle it,")
    print("OR I can patch benchmark.py to use this cache dictionary.")
    
    # We will simply print the dictionary to be pasted or we can modify the file.
    # User asked to "cache the best move into benchmark".
    # I will verify the output format first.
    
    print("\nCACHED_BEST_MOVES = {")
    for name, moves in cached_data.items():
        # format as string list
        moves_str = ", ".join([f"'{m}'" for m in moves])
        print(f"    '{name}': [{moves_str}],")
    print("}")

if __name__ == "__main__":
    generate_cache()
