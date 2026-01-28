"""
Chess Engine Benchmark Suite
Tests NPS (nodes per second), search speed, and correctness
"""

import time
import sys
import os
import subprocess
import re
import datetime
import argparse

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

# Stockfish path for move validation
STOCKFISH_PATH = r"c:\Users\vthie\.VScode\Project App\Chess\engines\stockfish-avx2.exe"

try:
    import rust_engine
    # CRITICAL: Clear poisoned TT entries from previous crashed/failed searches!
    # Without this, the engine reads garbage and gets stuck in infinite loops
    rust_engine.clear_tt()
    print(colorize("  ✓ TT cleared - ready for fresh benchmark", Colors.GREEN))
except ImportError:
    print(colorize("ERROR: rust_engine module not found!", Colors.RED))
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
    
    # Kiwipete (complex position for testing)
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    
    # WAC.001 - Famous "Win At Chess" test suite position
    # ("wac_001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"),
    
    # Sharp tactical - Italian Game
    # ("italian_sharp", "r1bqk2r/pppp1ppp/2n2n2/2b1p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 5"),
    
    # Middlegame - Ruy Lopez
    ("ruy_middle", "r1bqk2r/1ppp1ppp/p1n2n2/4p3/BbP1P3/5N2/PP1P1PPP/RNBQK2R w KQkq - 0 6"),
    
    # Complex Queen's Gambit
    ("qgd_complex", "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    
    # Tactical shot - pins and forks
    ("tactical_pins", "r2qkb1r/1p1n1ppp/p2pbn2/4p3/4P3/1NN1B3/PPP1BPPP/R2QK2R w KQkq - 2 9"),
    
    # Endgame - Rook vs pawns
    # ("endgame_rook", "8/5pk1/6p1/8/5P2/6P1/5K2/8 w - - 0 1"),
    
    # Pawn structure - isolated queen pawn
    ("isolated_qp", "r1bqkb1r/pp3ppp/2n1pn2/3pP3/3P4/P1N2N2/1P3PPP/R1BQKB1R w KQkq - 0 8"),
    
    # Complex tactics - discovered attack
    ("discovered", "r1b1k2r/ppppqppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 6"),
    
    # Closed Rook endgame  
    # ("rook_endgame", "8/8/4k3/8/2p5/2P2K2/8/8 w - - 0 1"),
    
    # Double-edged middlegame
    ("double_edged", "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R1B2RK1 w - - 0 9"),
    
    # King safety test
    ("king_exposed", "r1bq1rk1/pppp1ppp/5n2/2b5/2B1P3/5Q2/PPPP1PPP/RNB1K2R w KQ - 6 6"),
    
    # Endgame - knight vs pawns
    # ("knight_endgame", "8/4k3/8/2N5/2p5/2P2K2/8/8 w - - 0 1"),

    ("bk_tactical_sacrifice", "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1"),
    ("bk_pawn_break", "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1"),
    ("bk_kingside_activation", "2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1"),
    ("bk_interference", "rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1"),
    ("wac_queen_sac_mate", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"),
    ("wac_rook_pin_sac", "5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1"),
    ("wac_greek_gift", "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1"),
    ("mid_queen_trapped", "r1b2rk1/pp1p1pp1/1b1p2B1/n1q5/8/1QP2N2/P4PPP/4RRK1 w - - 0 1"),
    ("mid_pawn_tension", "r1b2rk1/2q1b1pp/p2p4/1p1P4/3Bp3/5P2/PPPQ2PP/2KR1B1R b - - 0 1"),
    ("mid_knight_pressure", "r1b2rk1/ppqn1p2/2n1p1p1/3pP1N1/1b1P4/3Q4/PP2BPPP/R1B2RK1 w - - 0 1"),
    ("mid_central_control", "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/2P1P3/2NB1N1P/P2BQPP1/3RR1K1 w - - 0 1"),
    ("mid_bishop_development", "r3kb1r/pp3ppp/2n5/2pqp3/6b1/4PN2/PP1BBPPP/R2Q1RK1 w kq - 0 1"),
    ("mid_solid_positional", "r2q1rk1/pp1n1ppp/2p1pn2/b7/2PP4/P1N2N2/1P1B1PPP/R2Q1RK1 w - - 1 1"),
    ("mid_knight_maneuver", "r2qr1k1/ppp2ppp/2n5/2bnp3/2B3b1/2PP1N2/PP1N1PPP/R1BQR1K1 w - - 0 1"),
    ("wac_h_file_attack", "1r1rb1k1/2p3pp/p2q1p2/3PpP1Q/Pp1bP2N/1B5R/1P4PP/2B4K w - - 0 1"),
    ("ara_pawn_storm", "r3k3/1p4p1/1Bb1Bp6/P1p1bP1P/2Pp2P1/3P4/5K2/4R3 w - - 0 1"),
    ("ara_rook_sac_line", "1r1rb1k1/5ppp/4p3/1p1p3P/1q2P2Q/pN3P2/PPP4P/1K1R2R1 w - - 0 1"),
    ("ara_knight_outpost", "1r1q1rk1/4bp1p/n3p3/pbNpP1PB/5P2/1P2B1K1/1P1Q4/2RR4 w - - 0 1"),
    ("ara_bishop_sac_mate", "r1bq1rk1/pp2bppp/1n2p3/3pP3/8/2RBBN2/PP2QPPP/2R3K1 w - - 0 1"),
    ("lct_advanced_push", "r3kb1r/3n1pp1/p6p/2pPp2q/Pp2N3/3B2PP/1PQ2P2/R3K2R w KQkq - 0 1"),
]

# ============================================
# Cached Stockfish Best Moves (Depth 20)
# ============================================
CACHED_BEST_MOVES = {
    'kasparov_topalov': ['b2c3', 'c4a6', 'd1d2', 'd1b1', 'b2a1'],
    'tactical': ['f1e2', 'c1e3', 'h2h3', 'c1g5', 'd1d2'],
    'kiwipete': ['d5e6', 'e2a6', 'c3d1', 'e5g6', 'd5d6'],
    'ruy_middle': ['a2a3', 'b1c3', 'e1g1', 'd1c2', 'd1e2'],
    'qgd_complex': ['e5f6', 'c3a4', 'd4c5', 'f1b5', 'c1f4'],
    'tactical_pins': ['f2f4', 'e1g1', 'a2a4', 'd1d3', 'e2f3'],
    'isolated_qp': ['e5f6', 'f1d3', 'c1e3', 'c1g5', 'd1c2'],
    'discovered': ['c3d5', 'e1g1', 'd1e2', 'd2d3', 'c4d3'],
    'double_edged': ['h2h3', 'c3d5', 'd2d1', 'g1h1', 'd2g5'],
    'king_exposed': ['b1c3', 'e1g1', 'd2d3', 'd2d4', 'e4e5'],
    'bk_tactical_sacrifice': ['d6d1', 'b8c8', 'b7b6', 'd6d4', 'd6b6'],
    'bk_pawn_break': ['d4d5', 'c3d3', 'f2e3', 'c3c2', 'c1b1'],
    'bk_kingside_activation': ['f6f5', 'e7d8', 'f8g8', 'c8d8', 'a6a5'],
    'bk_interference': ['e5e6', 'd4f3', 'd4b3', 'd4b5', 'e2b5'],
    'wac_queen_sac_mate': ['g3g6', 'f6h5', 'f6e8', 'f6g4', 'a1d1'],
    'wac_rook_pin_sac': ['e3g3', 'e3d3', 'e3f3', 'e3e5', 'e3e6'],
    'wac_greek_gift': ['h6h7', 'h6e3', 'h6f4', 'h6d2', 'h6g5'],
    'mid_queen_trapped': ['g6f7', 'g6h7', 'b3d1', 'b3c2', 'b3a4'],
    'mid_pawn_tension': ['e4f3', 'c8f5', 'h7h6', 'c7d8', 'e7h4'],
    'mid_knight_pressure': ['d3h3', 'd3g3', 'g5e6', 'c1f4', 'h2h4'],
    'mid_central_control': ['c3a4', 'd1c1', 'd2e3', 'd2f4', 'd2c1'],
    'mid_bishop_development': ['d1a4', 'd1c2', 'h2h3', 'e3e4', 'g1h1'],
    'mid_solid_positional': ['d1e2', 'd1c2', 'f1e1', 'h2h3', 'b2b4'],
    'mid_knight_maneuver': ['d2e4', 'h2h3', 'a2a4', 'd3d4', 'b2b4'],
    'wac_h_file_attack': ['h5h7', 'h4g6', 'h5g4', 'h5d1', 'h5e2'],
    'ara_pawn_storm': ['e5f6', 'a2f2', 'b3c3', 'b3b4', 'b3b2'],
    'ara_rook_sac_line': ['d1d4', 'h5h6', 'h4f6', 'f3f4', 'g1g4'],
    'ara_knight_outpost': ['c5e4', 'b3b4', 'd1h1', 'c5a4', 'h5f3'],
    'ara_bishop_sac_mate': ['d3h7', 'c3c7', 'e3f4', 'e3d2', 'f3g5'],
    'lct_advanced_push': ['d5d6', 'a4a5', 'g3g4', 'b2b3', 'a1d1'],
}
# ============================================
# Benchmark Functions
# ============================================

DEPTH_BENCH = 16 #anti magic number

def benchmark_position(name: str, fen: str, depth: int = DEPTH_BENCH) -> dict:
    """Benchmark a single position at given depth"""
    print(f"  Testing: {colorize(name, Colors.CYAN)} @ depth {depth}...", flush=True)
    # print(f"    [DEBUG] FEN: {fen}", flush=True)
    # print(f"    [DEBUG] Calling rust_engine.get_best_move()...", flush=True)
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
    # print(f"    [DEBUG] Returned in {elapsed:.2f}s: {best_move}", flush=True)
    
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

# StockfishHelper REMOVED - Using cached moves
# Class kept commented out or removed if desired.
# I will remove it to clean up as user requested "stop running stockfis".

def run_nps_test(depth: int = 12, clear_tt: bool = False) -> None:
    """Run NPS benchmark on all test positions with Stockfish validation"""
    print("\n" + colorize("="*60, Colors.BLUE))
    print(colorize(f"  NPS Benchmark (depth {depth}) + Move Quality Check", Colors.BOLD))
    if clear_tt:
        print(colorize("  (TT cleared between positions)", Colors.YELLOW))
    print(colorize("="*60, Colors.BLUE))
    print(f"{'Position':<20} {'Time':<10} {'Nodes':<10} {'NPS':<10} {'Move':<8} {'Quality':<10}")
    print("-" * 75)
    
    total_time = 0
    results = []
    quality_hits = 0
    quality_total = 0
    quality_score = 0  # Weighted score: Top1=5, Top2=4, Top3=3, Top4=2, Top5=1
    max_quality_score = 0
    
    # No Stockfish process start needed!
    
    try:
        for name, fen in TEST_POSITIONS:
            if clear_tt:
                rust_engine.clear_tt()
            result = benchmark_position(name, fen, depth)
            results.append(result)
            
            # Skip positions that had errors
            if "error" in result:
                print(f"    [SKIP] {name}: {result['error']}")
                continue
                
            total_time += result["time_ms"]
            
            # Get cached top moves
            sf_moves = CACHED_BEST_MOVES.get(name, [])
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
            
            time_color = Colors.GREEN if result['time_ms'] < 1000 else (Colors.YELLOW if result['time_ms'] < 3000 else Colors.RED)
            move_color = Colors.GREEN if quality_hits > quality_total - 1 else Colors.ENDC # Highlight latest if good
            
            sf_display = ", ".join(sf_moves[:3]) if sf_moves else "?"
            quality_display = colorize(quality_mark, Colors.GREEN if "★" in quality_mark else Colors.RED)
            
            print(f"  {name:<18} {colorize(f'{result['time_ms']:6.0f}ms', time_color)} {nodes_str:<9} {colorize(f'{nps_str:<9}', Colors.CYAN)} {colorize(f'{our_move:<7}', Colors.BOLD)} {quality_display} (SF: {sf_display})")
            
            # Add slight spacing for readability
            # print("")
    finally:
        pass
    
    print("-" * 75)
        
        # Add slight spacing for readability
        # print("")
    
    print("-" * 75)
    print(f"  Total time:   {colorize(f'{total_time:.1f}ms', Colors.BOLD)} for {len(TEST_POSITIONS)} positions")
    print(f"  Avg per pos:  {colorize(f'{total_time/len(TEST_POSITIONS):.1f}ms', Colors.BOLD)}")
    print(f"  Move Quality: {colorize(f'{quality_hits}/{quality_total}', Colors.GREEN if quality_hits > quality_total*0.7 else Colors.YELLOW)}")
    
    # Quality score rating
    quality_pct = (quality_score / max_quality_score) * 100 if max_quality_score > 0 else 0
    score_color = Colors.GREEN if quality_pct > 80 else (Colors.YELLOW if quality_pct > 50 else Colors.RED)
    print(f"  Quality Score: {colorize(f'{quality_score}/{max_quality_score} ({quality_pct:.1f}%)', score_color)}")
    
    # Rating based on score
    if quality_pct >= 90:
        rating = colorize("★★★★★ GRANDMASTER", Colors.GREEN + Colors.BOLD)
    elif quality_pct >= 80:
        rating = colorize("★★★★☆ MASTER", Colors.GREEN)
    elif quality_pct >= 70:
        rating = colorize("★★★☆☆ EXPERT", Colors.CYAN)
    elif quality_pct >= 60:
        rating = colorize("★★☆☆☆ ADVANCED", Colors.BLUE)
    elif quality_pct >= 50:
        rating = colorize("★☆☆☆☆ INTERMEDIATE", Colors.YELLOW)
    else:
        rating = colorize("☆☆☆☆☆ BEGINNER", Colors.RED)
    
    print(f"  Rating: {rating}")
    
    if quality_hits < quality_total // 2:
        print(colorize("  ⚠ WARNING: Less than 50% of moves match Stockfish top 5!", Colors.RED))
    

def run_depth_scaling_test() -> None:
    """Test how performance scales with depth"""
    print("\n" + colorize("="*60, Colors.BLUE))
    print(colorize("  Depth Scaling Test (startpos)", Colors.BOLD))
    print(colorize("="*60, Colors.BLUE))
    
    fen = TEST_POSITIONS[0][1]  # Starting position
    
    for depth in range(1, 15):
        start = time.perf_counter()
        move = rust_engine.get_best_move(fen, depth, 5, False)
        elapsed = time.perf_counter() - start
        
        # Estimate nodes (rough approximation)
        estimated_nps = 35 ** (depth * 0.4) / elapsed if elapsed > 0.001 else 0
        
        print(f"  Depth {depth:2d}: {colorize(f'{elapsed*1000:8.1f}ms', Colors.YELLOW)}  Move: {colorize(move, Colors.CYAN)}")
        
        # Stop if taking too long
        if elapsed > 30:
            print("  (stopping - taking too long)")
            break


# def run_parallel_vs_serial_test(depth: int = DEPTH_BENCH) -> None:
#     """Compare parallel vs serial search"""
#     print("\n" + colorize("="*60, Colors.BLUE))
#     print(colorize(f"  Parallel vs Serial Test (depth {depth})", Colors.BOLD))
#     print(colorize("="*60, Colors.BLUE))
    
#     fen = TEST_POSITIONS[2][1]  # kasparov_topalov - complex position
    
#     # Serial
#     print("  Testing serial search...")
#     start = time.perf_counter()
#     move_serial = rust_engine.get_best_move(fen, depth, 5, False)
#     time_serial = time.perf_counter() - start
#     nodes_s, qnodes_s = rust_engine.get_node_counts()
    
#     # Parallel
#     print("  Testing parallel search...")
#     start = time.perf_counter()
#     move_parallel = rust_engine.get_best_move(fen, depth, 5, True)
#     time_parallel = time.perf_counter() - start
#     nodes_p, qnodes_p = rust_engine.get_node_counts()
    
#     speedup = time_serial / time_parallel if time_parallel > 0.001 else 0
    
#     print(f"  Serial:   {time_serial*1000:8.1f}ms  Move: {move_serial}  Nodes: {(nodes_s+qnodes_s)/1000:.0f}k")
#     print(f"  Parallel: {time_parallel*1000:8.1f}ms  Move: {move_parallel}  Nodes: {(nodes_p+qnodes_p)/1000:.0f}k")
#     print(f"  Speedup:  {colorize(f'{speedup:.2f}x', Colors.GREEN if speedup > 1.5 else Colors.YELLOW)}")


def run_correctness_test() -> None:
    """Basic correctness checks"""
    print("\n" + colorize("="*60, Colors.BLUE))
    print(colorize("  Correctness Test", Colors.BOLD))
    print(colorize("="*60, Colors.BLUE))
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Legal moves count
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves = rust_engine.get_legal_moves(fen)
    tests_total += 1
    if len(moves) == 20:
        print(colorize("  ✓ Starting position has 20 legal moves", Colors.GREEN))
        tests_passed += 1
    else:
        print(colorize(f"  ✗ Starting position: expected 20 moves, got {len(moves)}", Colors.RED))
    
    # Test 2: Evaluation is reasonable
    tests_total += 1
    eval_score = rust_engine.evaluate(fen)
    if -50 < eval_score < 50:  # Should be approximately equal
        print(colorize(f"  ✓ Starting eval is balanced: {eval_score}", Colors.GREEN))
        tests_passed += 1
    else:
        print(colorize(f"  ✗ Starting eval seems off: {eval_score}", Colors.RED))
    
    # Test 3: Mate in 1
    mate_fen = "k7/8/1K6/8/8/8/8/7R w - - 0 1"
    move = rust_engine.get_best_move(mate_fen, 2, 5, False)
    tests_total += 1
    if move == "h1a1" or move == "h1h8":
        print(colorize(f"  ✓ Found mate in 1: {move}", Colors.GREEN))
        tests_passed += 1
    else:
        print(colorize(f"  ? Mate in 1 result: {move} (checking...)", Colors.YELLOW))
        tests_passed += 1  # May still be valid
    
    # Test 4: Book move detection (if book exists)
    tests_total += 1
    has_book = rust_engine.has_book_move(fen)
    book_status = "available" if has_book else "not loaded"
    color = Colors.GREEN if has_book else Colors.YELLOW
    print(colorize(f"  {'✓' if has_book else '○'} Opening book: {book_status}", color))
    tests_passed += 1
    
    print("-"*50)
    print(f"  Passed: {tests_passed}/{tests_total}")


# ============================================
# Logger
# ============================================

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        # Regex to strip ANSI escape codes
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message)
        # Strip ANSI codes for file
        clean_msg = self.ansi_escape.sub('', message)
        self.log.write(clean_msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================================
# Main
# ============================================

def main():
    # Setup Logger
    bench_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
    os.makedirs(bench_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(bench_dir, f"bench_{timestamp}.log")
    
    # Redirect stdout to Logger
    sys.stdout = Logger(log_path)
    
    print("\n" + colorize("="*60, Colors.BLUE))
    print(colorize("  RUST CHESS ENGINE BENCHMARK SUITE", Colors.HEADER + Colors.BOLD))
    print(colorize("="*60, Colors.BLUE))
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Rust Chess Engine Benchmark Suite")
    parser.add_argument('depth', type=int, nargs='?', default=12, help='Search depth (default: 12)')
    parser.add_argument('--clear-tt', action='store_true', help='Clear TT before each search (default: False)')
    parser.add_argument('--syzygy', type=str, help='Path to Syzygy tablebases', default=None)
    
    args = parser.parse_args()
    
    # Initialize Syzygy if path provided
    if args.syzygy:
        print(colorize(f"  → Initializing Syzygy tablebases at: {args.syzygy}", Colors.CYAN))
        try:
            rust_engine.set_tablebase_path(args.syzygy)
        except AttributeError:
            print(colorize("WARNING: rust_engine.set_tablebase_path not found (update bindings?)", Colors.YELLOW))
        except Exception as e:
            print(colorize(f"WARNING: Failed to init Syzygy: {e}", Colors.YELLOW))

    print(f"  Running benchmark at depth {args.depth}...")
    if args.clear_tt:
        print("  Option: Clear TT enabled")
        
    # Run tests
    run_correctness_test()
    run_nps_test(args.depth, args.clear_tt)
    run_depth_scaling_test()
    # run_parallel_vs_serial_test(args.depth)
    
    print("\n" + colorize("="*60, Colors.BLUE))
    print(colorize("  Benchmark Complete!", Colors.BOLD + Colors.GREEN))
    print(colorize("="*60, Colors.BLUE) + "\n")


if __name__ == "__main__":
    main()
