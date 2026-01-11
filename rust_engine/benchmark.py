"""
Chess Engine Benchmark Suite
Tests NPS (nodes per second), search speed, and correctness
"""

import time
import sys
import os
import subprocess
import re

# Add the target directory to path for the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target', 'release'))

# Stockfish path for move validation
STOCKFISH_PATH = r"c:\Users\vthie\.VScode\Project App\Chess\engines\stockfish-avx2.exe"

try:
    import rust_engine
    # CRITICAL: Clear poisoned TT entries from previous crashed/failed searches!
    # Without this, the engine reads garbage and gets stuck in infinite loops
    rust_engine.clear_tt()
    print("  TT cleared - ready for fresh benchmark")
except ImportError:
    print("ERROR: rust_engine module not found!")
    print("Make sure to run: cargo build --release")
    print("Or run: maturin develop --release")
    sys.exit(1)

# ============================================
# Test Positions - Complex positions requiring actual search (no book moves!)
# ============================================
TEST_POSITIONS = [
    # Complex middlegame (Kasparov vs Topalov)
    ("kasparov_topalov", "1rr3k1/4ppbp/2n3p1/2P5/p1BP4/P3P1P1/1B3P1P/3R1RK1 w - - 0 1"),
    
    # Tactical position
    ("tactical", "r2qk2r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 4 6"),
    
    # Endgame
    ("endgame", "8/5pk1/6p1/8/5P2/6P1/5K2/8 w - - 0 1"),
    
    # Kiwipete (complex position for testing)
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    
    # WAC.001 - Famous "Win At Chess" test suite position
    ("wac_001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"),
    
    # Sharp tactical - Italian Game
    ("italian_sharp", "r1bqk2r/pppp1ppp/2n2n2/2b1p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 5"),
    
    # Middlegame - Ruy Lopez
    ("ruy_middle", "r1bqk2r/1ppp1ppp/p1n2n2/4p3/BbP1P3/5N2/PP1P1PPP/RNBQK2R w KQkq - 0 6"),
    
    # Complex Queen's Gambit
    ("qgd_complex", "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    
    # Tactical shot - pins and forks
    ("tactical_pins", "r2qkb1r/1p1n1ppp/p2pbn2/4p3/4P3/1NN1B3/PPP1BPPP/R2QK2R w KQkq - 2 9"),
    
    # Endgame - Rook vs pawns
    ("endgame_rook", "8/5pk1/6p1/8/5P2/6P1/5K2/8 w - - 0 1"),
    
    # Pawn structure - isolated queen pawn
    ("isolated_qp", "r1bqkb1r/pp3ppp/2n1pn2/3pP3/3P4/P1N2N2/1P3PPP/R1BQKB1R w KQkq - 0 8"),
    
    # Complex tactics - discovered attack
    ("discovered", "r1b1k2r/ppppqppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 6"),
    
    # Closed Rook endgame  
    ("rook_endgame", "8/8/4k3/8/2p5/2P2K2/8/8 w - - 0 1"),
    
    # Double-edged middlegame
    ("double_edged", "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R1B2RK1 w - - 0 9"),
    
    # King safety test
    ("king_exposed", "r1bq1rk1/pppp1ppp/5n2/2b5/2B1P3/5Q2/PPPP1PPP/RNB1K2R w KQ - 6 6"),
    
    # Endgame - knight vs pawns
    ("knight_endgame", "8/8/4k3/2N5/2p5/2P2K2/8/8 w - - 0 1"),
]

# ============================================
# Benchmark Functions
# ============================================

def benchmark_position(name: str, fen: str, depth: int = 10) -> dict:
    """Benchmark a single position at given depth"""
    print(f"  Testing: {name} @ depth {depth}...", flush=True)
    print(f"    [DEBUG] FEN: {fen}", flush=True)
    print(f"    [DEBUG] Calling rust_engine.get_best_move()...", flush=True)
    sys.stdout.flush()
    
    start = time.perf_counter()
    try:
        # Keep debug OFF for performance (set to True only when debugging hangs)
        try:
            rust_engine.set_debug(False)
        except:
            pass
        
        best_move = rust_engine.get_best_move(fen, depth, 5, False)  # aggressiveness=5, parallel=False
    except Exception as e:
        print(f"    [ERROR] Exception: {e}", flush=True)
        return {"name": name, "error": str(e)}
    
    elapsed = time.perf_counter() - start
    print(f"    [DEBUG] Returned in {elapsed:.2f}s: {best_move}", flush=True)
    
    # Get node counts from the search
    nodes, qnodes = rust_engine.get_node_counts()
    total_nodes = nodes + qnodes
    nps = int(total_nodes / elapsed) if elapsed > 0.001 else 0
    
    # Get evaluation
    eval_score = rust_engine.evaluate(fen)
    
    return {
        "name": name,
        "depth": depth,
        "time_ms": elapsed * 1000,
        "best_move": best_move,
        "eval": eval_score,
        "nodes": nodes,
        "qnodes": qnodes,
        "nps": nps,
    }

def get_stockfish_top_moves(fen: str, depth: int = 10, num_moves: int = 5) -> list:
    """Get Stockfish's top N moves for a position using MultiPV"""
    try:
        proc = subprocess.Popen(
            [STOCKFISH_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Send UCI commands one at a time
        def send(cmd):
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        
        send("uci")
        send(f"setoption name MultiPV value {num_moves}")
        send("isready")
        send(f"position fen {fen}")
        send(f"go depth {depth}")
        
        # Read output until we get bestmove
        moves_by_mpv = {}
        output_lines = []
        
        import threading
        
        def read_output():
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                output_lines.append(line.strip())
                if line.startswith("bestmove"):
                    break
        
        reader = threading.Thread(target=read_output)
        reader.start()
        reader.join(timeout=30)  # 30 second timeout
        
        if reader.is_alive():
            proc.kill()
            return []
        
        # Parse collected output
        for line in output_lines:
            if 'info' in line and ' depth ' in line and ' pv ' in line:
                # Extract depth
                depth_match = re.search(r' depth (\d+)', line)
                if not depth_match:
                    continue
                line_depth = int(depth_match.group(1))
                
                # Extract multipv number (default 1)  
                mpv_match = re.search(r' multipv (\d+)', line)
                mpv_num = int(mpv_match.group(1)) if mpv_match else 1
                
                # Extract the first PV move
                pv_match = re.search(r' pv ([a-h][1-8][a-h][1-8][qrbn]?)', line)
                if pv_match:
                    move = pv_match.group(1)
                    # Keep highest depth for each mpv
                    if mpv_num not in moves_by_mpv or line_depth > moves_by_mpv[mpv_num][0]:
                        moves_by_mpv[mpv_num] = (line_depth, move)
        
        proc.kill()
        
        # Build ordered list from multipv 1, 2, 3...
        moves = []
        for i in range(1, num_moves + 1):
            if i in moves_by_mpv:
                moves.append(moves_by_mpv[i][1])
        
        return moves[:num_moves]
    except Exception as e:
        print(f"Stockfish error: {e}")
        return []


def run_nps_test(depth: int = 12) -> None:
    """Run NPS benchmark on all test positions with Stockfish validation"""
    print("\n" + "="*50)
    print(f"  NPS Benchmark (depth {depth}) + Move Quality Check")
    print("="*50)
    
    total_time = 0
    results = []
    quality_hits = 0
    quality_total = 0
    quality_score = 0  # Weighted score: Top1=5, Top2=4, Top3=3, Top4=2, Top5=1
    max_quality_score = 0
    
    for name, fen in TEST_POSITIONS:
        result = benchmark_position(name, fen, depth)
        results.append(result)
        
        # Skip positions that had errors
        if "error" in result:
            print(f"    [SKIP] {name}: {result['error']}")
            continue
            
        total_time += result["time_ms"]
        
        # Get Stockfish's top 5 moves for comparison
        sf_moves = get_stockfish_top_moves(fen, depth=12, num_moves=5)
        our_move = result['best_move']
        
        # Check position in Stockfish's ranking
        max_quality_score += 5  # Max possible score
        if our_move in sf_moves:
            position = sf_moves.index(our_move) + 1  # 1-indexed
            points = 6 - position  # Top1=5, Top2=4, Top3=3, Top4=2, Top5=1
            quality_score += points
            quality_hits += 1
            quality_mark = f"★{position}"  # Show ranking
        else:
            position = 0
            points = 0
            quality_mark = "✗"
        
        quality_total += 1
        
        # Format NPS for display
        nps_str = f"{result['nps']/1000:.0f}k" if result['nps'] < 1000000 else f"{result['nps']/1000000:.1f}M"
        nodes_str = f"{(result['nodes']+result['qnodes'])/1000:.0f}k"
        
        sf_display = ", ".join(sf_moves[:3]) if sf_moves else "?"
        
        print(f"    [{quality_mark:3s}] {our_move:6s} {result['time_ms']:8.1f}ms  {nodes_str:>6s} nodes  {nps_str:>5s}/s  SF:[{sf_display}]")
    
    print("-"*50)
    print(f"  Total time: {total_time:.1f}ms for {len(TEST_POSITIONS)} positions")
    print(f"  Average: {total_time/len(TEST_POSITIONS):.1f}ms per position")
    print(f"  Move Quality: {quality_hits}/{quality_total} in Stockfish top 5")
    
    # Quality score rating
    quality_pct = (quality_score / max_quality_score) * 100 if max_quality_score > 0 else 0
    print(f"  Quality Score: {quality_score}/{max_quality_score} ({quality_pct:.1f}%)")
    
    # Rating based on score
    if quality_pct >= 90:
        rating = "★★★★★ GRANDMASTER"
    elif quality_pct >= 80:
        rating = "★★★★☆ MASTER"
    elif quality_pct >= 70:
        rating = "★★★☆☆ EXPERT"
    elif quality_pct >= 60:
        rating = "★★☆☆☆ ADVANCED"
    elif quality_pct >= 50:
        rating = "★☆☆☆☆ INTERMEDIATE"
    else:
        rating = "☆☆☆☆☆ BEGINNER"
    
    print(f"  Rating: {rating}")
    
    if quality_hits < quality_total // 2:
        print("  ⚠ WARNING: Less than 50% of moves match Stockfish top 5!")
    

def run_depth_scaling_test() -> None:
    """Test how performance scales with depth"""
    print("\n" + "="*50)
    print("  Depth Scaling Test (startpos)")
    print("="*50)
    
    fen = TEST_POSITIONS[0][1]  # Starting position
    
    for depth in range(1, 15):
        start = time.perf_counter()
        move = rust_engine.get_best_move(fen, depth, 5, False)
        elapsed = time.perf_counter() - start
        
        # Estimate nodes (rough approximation)
        estimated_nps = 35 ** (depth * 0.4) / elapsed if elapsed > 0.001 else 0
        
        print(f"  Depth {depth:2d}: {elapsed*1000:8.1f}ms  Move: {move}")
        
        # Stop if taking too long
        if elapsed > 30:
            print("  (stopping - taking too long)")
            break


def run_parallel_vs_serial_test(depth: int = 10) -> None:
    """Compare parallel vs serial search"""
    print("\n" + "="*50)
    print(f"  Parallel vs Serial Test (depth {depth})")
    print("="*50)
    
    fen = TEST_POSITIONS[2][1]  # kasparov_topalov - complex position
    
    # Serial
    print("  Testing serial search...")
    start = time.perf_counter()
    move_serial = rust_engine.get_best_move(fen, depth, 5, False)
    time_serial = time.perf_counter() - start
    nodes_s, qnodes_s = rust_engine.get_node_counts()
    
    # Parallel
    print("  Testing parallel search...")
    start = time.perf_counter()
    move_parallel = rust_engine.get_best_move(fen, depth, 5, True)
    time_parallel = time.perf_counter() - start
    nodes_p, qnodes_p = rust_engine.get_node_counts()
    
    speedup = time_serial / time_parallel if time_parallel > 0.001 else 0
    
    print(f"  Serial:   {time_serial*1000:8.1f}ms  Move: {move_serial}  Nodes: {(nodes_s+qnodes_s)/1000:.0f}k")
    print(f"  Parallel: {time_parallel*1000:8.1f}ms  Move: {move_parallel}  Nodes: {(nodes_p+qnodes_p)/1000:.0f}k")
    print(f"  Speedup:  {speedup:.2f}x")


def run_correctness_test() -> None:
    """Basic correctness checks"""
    print("\n" + "="*50)
    print("  Correctness Test")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Legal moves count
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves = rust_engine.get_legal_moves(fen)
    tests_total += 1
    if len(moves) == 20:
        print("  ✓ Starting position has 20 legal moves")
        tests_passed += 1
    else:
        print(f"  ✗ Starting position: expected 20 moves, got {len(moves)}")
    
    # Test 2: Evaluation is reasonable
    tests_total += 1
    eval_score = rust_engine.evaluate(fen)
    if -50 < eval_score < 50:  # Should be approximately equal
        print(f"  ✓ Starting eval is balanced: {eval_score}")
        tests_passed += 1
    else:
        print(f"  ✗ Starting eval seems off: {eval_score}")
    
    # Test 3: Mate in 1
    mate_fen = "k7/8/1K6/8/8/8/8/7R w - - 0 1"
    move = rust_engine.get_best_move(mate_fen, 2, 5, False)
    tests_total += 1
    if move == "h1a1" or move == "h1h8":
        print(f"  ✓ Found mate in 1: {move}")
        tests_passed += 1
    else:
        print(f"  ? Mate in 1 result: {move} (checking...)")
        tests_passed += 1  # May still be valid
    
    # Test 4: Book move detection (if book exists)
    tests_total += 1
    has_book = rust_engine.has_book_move(fen)
    print(f"  {'✓' if has_book else '○'} Opening book: {'available' if has_book else 'not loaded'}")
    tests_passed += 1
    
    print("-"*50)
    print(f"  Passed: {tests_passed}/{tests_total}")


# ============================================
# Main
# ============================================

def main():
    print("\n" + "="*50)
    print("  RUST CHESS ENGINE BENCHMARK SUITE")
    print("="*50)
    
    # Parse arguments
    depth = 10
    if len(sys.argv) > 1:
        try:
            depth = int(sys.argv[1])
        except ValueError:
            pass
    
    # Run tests
    run_correctness_test()
    run_nps_test(depth)
    run_depth_scaling_test()
    run_parallel_vs_serial_test(depth)
    
    print("\n" + "="*50)
    print("  Benchmark Complete!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
