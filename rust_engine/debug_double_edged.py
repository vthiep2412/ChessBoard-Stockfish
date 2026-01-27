import sys
import os
import time

# Add release build to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target', 'release'))

try:
    import rust_engine
except ImportError:
    print("rust_engine not found. Build with `cargo build --release`")
    sys.exit(1)

FEN = "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R1B2RK1 w - - 0 9"
DEPTHS = [12, 14, 16]

rust_engine.clear_tt()
rust_engine.set_debug(True) # Enable detailed search info

print(f"Debugging Position: {FEN}")

for depth in DEPTHS:
    print(f"\n{'='*40}")
    print(f"Running Depth {depth}...")
    print(f"{'='*40}")
    
    start = time.perf_counter()
    move = rust_engine.get_best_move(FEN, depth, 5, False)
    elapsed = time.perf_counter() - start
    
    eval_score = rust_engine.evaluate(FEN)
    print(f"\nDepth {depth} Result:")
    print(f"  Best Move: {move}")
    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Eval: {eval_score}")
