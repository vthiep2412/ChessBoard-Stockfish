import chess
import chess.engine
import sys
import os
import time

# STOCKFISH_PATH should be the one we just downloaded
STOCKFISH_PATH = "./stockfish.exe"

TEST_POSITIONS = [
    ("kasparov_topalov", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NB1/pPP2PPP/R3K2R b KQ - 0 1"),
    ("tactical", "r3k2r/pp1n1ppp/2p5/4Pb2/2B2P2/2N5/PPP3PP/R3K2R w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("wac_001", "2rr3k/pp3pp1/1nnqbN1p/3p4/2pP4/2P3N1/PPBQ1PPP/R3R1K1 w - - 0 1"),
    ("italian_sharp", "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"),
    ("ruy_middle", "r1bq1rk1/2p1bppp/p2p1n2/1p2p3/3nP3/1BPP1N2/PP3PPP/RNBQ1RK1 w - - 0 1"),
    ("qgd_complex", "rnbq1rk1/pp3ppp/4pn2/2bp4/2P5/2N2NP1/PP2PPBP/R1BQK2R w KQ - 0 1"),
    ("tactical_pins", "r1bq1rk1/ppp2ppp/2n5/3pP3/2BP4/5N2/P2Q1PPP/R4RK1 w - - 0 1"),
    ("endgame_rook", "8/8/8/6p1/5k2/4R3/5P1P/5K2 w - - 0 1"),
    ("isolated_qp", "r1b1qrk1/pp2bppp/2n1pn2/8/2BP4/2N2N2/PP2QPPP/R1B2RK1 w - - 0 1"),
    ("discovered", "rnbqk2r/ppp2ppp/4pn2/3p4/1bPP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 1"),
    ("rook_endgame", "8/8/5k2/8/8/5P2/4R3/5K2 w - - 0 1"),
    ("double_edged", "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R1B2RK1 w - - 0 9"),
    ("king_exposed", "r1bq1rk1/pp3ppp/2n1p3/2pp4/2PP4/P1P1PN2/2Q2PPP/R3KB1R w KQ - 0 1"),
    ("knight_endgame", "8/8/5k2/5p2/8/5P2/4N3/5K2 w - - 0 1"),
]

def run_benchmark():
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Error: Stockfish binary not found at {STOCKFISH_PATH}")
        return

    print("Starting Stockfish (HCE Mode - NNUE Disabled)...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Failed to start engine: {e}")
        return

    # Disable NNUE to test "Base Strength" (Hand Crafted Evaluation)
    try:
        engine.configure({"Use NNUE": False})
        print("Confirmed: NNUE Disabled.")
    except Exception as e:
        print(f"Warning: Could not disable NNUE (maybe old version?): {e}")

    print(f"\n{'Position':<20} {'Time':<10} {'Nodes':<10} {'NPS':<10} {'Move':<8} {'Score':<10}")
    print("-" * 80)

    total_time = 0
    
    for name, fen in TEST_POSITIONS:
        board = chess.Board(fen)
        
        # Clear Hash to prevent pollution (Fair comparison with my Engine's clear_tt)
        engine.configure({"Clear Hash": None})
        
        start = time.perf_counter()
        # Depth 16 to match Rust engine target
        result = engine.play(board, chess.engine.Limit(depth=16))
        elapsed = time.perf_counter() - start
        
        # Get info (requires analysis, play doesn't return nodes/score easily in python-chess without info handler)
        # Re-running analysis for stats or just using time from play? 
        # Better: use analysis with limit.
        
        info = engine.analyse(board, chess.engine.Limit(depth=16))
        
        nodes = info.get("nodes", 0)
        nps = info.get("nps", 0)
        score = info.get("score", chess.engine.Cp(0))
        struct_score = score.white().score(mate_score=10000)
        
        # Move formatting
        best_move_uci = info.get("pv", [None])[0].uci() if info.get("pv") else "????"
        
        total_time += elapsed * 1000
        
        nps_str = f"{nps/1000000:.1f}M"
        nodes_str = f"{nodes/1000:.0f}k"
        
        print(f"{name:<20} {elapsed*1000:6.0f}ms {nodes_str:<10} {nps_str:<10} {best_move_uci:<8} {struct_score:<10}")

    engine.quit()
    print("-" * 80)
    print(f"Total Time: {total_time:.0f}ms")

if __name__ == "__main__":
    run_benchmark()
