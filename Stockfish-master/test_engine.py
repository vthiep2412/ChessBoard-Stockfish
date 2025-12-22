"""
Stockfish Performance Tester
============================
Compares baseline vs modified Stockfish builds.

Usage:
    python test_engine.py                    # Run all tests
    python test_engine.py --bench            # Bench only
    python test_engine.py --positions        # Position analysis only
    python test_engine.py --match            # Self-play match only
"""

import subprocess
import time
import os
import sys
import re
from dataclasses import dataclass
from typing import Optional
import statistics

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths to Stockfish executables
BASELINE_ENGINE = "./stockfish_baseline.exe"  # Rename your original build to this
MODIFIED_ENGINE = "./stockfish.exe"           # Your modified build

# Test positions (strategic and tactical mix)
TEST_POSITIONS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Sicilian Najdorf
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # Kings Indian Attack
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 4",
    # Endgame - Rook+Pawn
    "8/8/8/4k3/R7/4K3/4P3/8 w - - 0 1",
    # Tactical - Pin
    "r2qkb1r/ppp2ppp/2np1n2/4p3/2B1P1b1/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6",
    # Complex middlegame
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9",
    # Known tactical position (mate in 4)
    "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1",
    # Endgame - Bishop vs Knight
    "8/5k2/8/8/3B4/8/5K2/6n1 w - - 0 1",
]

ANALYSIS_DEPTH = 16  # Depth for position analysis
MATCH_GAMES = 10     # Number of self-play games
MATCH_TIME_MS = 500  # Time per move in self-play (milliseconds)


@dataclass
class BenchResult:
    nodes: int
    nps: int
    time_ms: int


@dataclass
class PositionResult:
    fen: str
    depth: int
    score: int
    best_move: str
    nodes: int
    time_ms: int


@dataclass
class MatchResult:
    wins: int
    losses: int
    draws: int
    
    @property
    def score(self) -> float:
        total = self.wins + self.losses + self.draws
        if total == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / total


# ==============================================================================
# ENGINE COMMUNICATION
# ==============================================================================

class Engine:
    def __init__(self, path: str):
        self.path = path
        self.process = None
    
    def start(self):
        self.process = subprocess.Popen(
            self.path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._send("uci")
        self._wait_for("uciok")
    
    def stop(self):
        if self.process:
            self._send("quit")
            self.process.wait(timeout=5)
            self.process = None
    
    def _send(self, cmd: str):
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
    
    def _read_line(self, timeout: float = 30.0) -> str:
        import select
        # Simple blocking read for Windows compatibility
        return self.process.stdout.readline().strip()
    
    def _wait_for(self, keyword: str, timeout: float = 60.0) -> list[str]:
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line()
            if line:
                lines.append(line)
                if keyword in line:
                    return lines
        raise TimeoutError(f"Timeout waiting for '{keyword}'")
    
    def bench(self) -> BenchResult:
        """Run the built-in bench command."""
        self._send("bench")
        lines = self._wait_for("Nodes/second", timeout=120)
        
        nodes = 0
        nps = 0
        time_ms = 0
        
        for line in lines:
            if "Nodes searched" in line:
                match = re.search(r"(\d+)", line.replace(",", ""))
                if match:
                    nodes = int(match.group(1))
            elif "Nodes/second" in line:
                match = re.search(r"(\d+)", line.replace(",", ""))
                if match:
                    nps = int(match.group(1))
        
        return BenchResult(nodes=nodes, nps=nps, time_ms=time_ms)
    
    def analyze(self, fen: str, depth: int) -> PositionResult:
        """Analyze a position to a given depth."""
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        
        lines = self._wait_for("bestmove")
        
        score = 0
        best_move = ""
        nodes = 0
        time_ms = 0
        
        for line in lines:
            if line.startswith("info") and f"depth {depth}" in line:
                # Parse score
                score_match = re.search(r"score cp (-?\d+)", line)
                mate_match = re.search(r"score mate (-?\d+)", line)
                if score_match:
                    score = int(score_match.group(1))
                elif mate_match:
                    mate_in = int(mate_match.group(1))
                    score = 10000 * (1 if mate_in > 0 else -1)
                
                # Parse nodes
                nodes_match = re.search(r"nodes (\d+)", line)
                if nodes_match:
                    nodes = int(nodes_match.group(1))
                
                # Parse time
                time_match = re.search(r"time (\d+)", line)
                if time_match:
                    time_ms = int(time_match.group(1))
            
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
        
        return PositionResult(
            fen=fen, depth=depth, score=score, 
            best_move=best_move, nodes=nodes, time_ms=time_ms
        )


# ==============================================================================
# TESTS
# ==============================================================================

def run_bench_test(baseline_path: str, modified_path: str) -> dict:
    """Compare bench performance between two engines."""
    print("\n" + "="*60)
    print("BENCH TEST")
    print("="*60)
    
    results = {}
    
    for name, path in [("Baseline", baseline_path), ("Modified", modified_path)]:
        if not os.path.exists(path):
            print(f"  ⚠️  {name} not found: {path}")
            continue
        
        print(f"\n  Testing {name}...")
        engine = Engine(path)
        try:
            engine.start()
            result = engine.bench()
            results[name.lower()] = result
            print(f"    Nodes: {result.nodes:,}")
            print(f"    NPS:   {result.nps:,}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        finally:
            engine.stop()
    
    # Compare
    if "baseline" in results and "modified" in results:
        baseline_nps = results["baseline"].nps
        modified_nps = results["modified"].nps
        diff = modified_nps - baseline_nps
        pct = (diff / baseline_nps) * 100 if baseline_nps else 0
        
        print(f"\n  📊 COMPARISON:")
        print(f"    NPS Difference: {diff:+,} ({pct:+.2f}%)")
        
        if pct > 0:
            print(f"    ✅ Modified is FASTER")
        elif pct < 0:
            print(f"    ❌ Modified is SLOWER")
        else:
            print(f"    ➖ No difference")
    
    return results


def run_position_test(baseline_path: str, modified_path: str) -> dict:
    """Analyze test positions and compare scores/depth."""
    print("\n" + "="*60)
    print("POSITION ANALYSIS TEST")
    print("="*60)
    print(f"  Depth: {ANALYSIS_DEPTH}")
    print(f"  Positions: {len(TEST_POSITIONS)}")
    
    results = {"baseline": [], "modified": []}
    
    for name, path in [("Baseline", baseline_path), ("Modified", modified_path)]:
        if not os.path.exists(path):
            print(f"\n  ⚠️  {name} not found: {path}")
            continue
        
        print(f"\n  Testing {name}...")
        engine = Engine(path)
        try:
            engine.start()
            for i, fen in enumerate(TEST_POSITIONS):
                result = engine.analyze(fen, ANALYSIS_DEPTH)
                results[name.lower()].append(result)
                print(f"    Position {i+1}: score={result.score:+d} move={result.best_move} nodes={result.nodes:,}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        finally:
            engine.stop()
    
    # Compare
    if results["baseline"] and results["modified"]:
        print(f"\n  📊 COMPARISON:")
        
        baseline_nodes = [r.nodes for r in results["baseline"]]
        modified_nodes = [r.nodes for r in results["modified"]]
        
        avg_baseline = statistics.mean(baseline_nodes)
        avg_modified = statistics.mean(modified_nodes)
        
        node_diff = avg_modified - avg_baseline
        node_pct = (node_diff / avg_baseline) * 100 if avg_baseline else 0
        
        print(f"    Avg Nodes - Baseline: {avg_baseline:,.0f}")
        print(f"    Avg Nodes - Modified: {avg_modified:,.0f}")
        print(f"    Difference: {node_diff:+,.0f} ({node_pct:+.2f}%)")
        
        # Check if scores differ (might indicate evaluation changes)
        score_diffs = []
        for b, m in zip(results["baseline"], results["modified"]):
            score_diffs.append(m.score - b.score)
        
        avg_score_diff = statistics.mean(score_diffs)
        print(f"    Avg Score Difference: {avg_score_diff:+.1f} cp")
    
    return results


def run_summary():
    """Print a summary of what files to test."""
    print("\n" + "="*60)
    print("STOCKFISH TESTING GUIDE")
    print("="*60)
    print("""
  SETUP:
  1. Copy your current stockfish.exe to stockfish_baseline.exe
  2. Make modifications to the source code
  3. Rebuild: make -j build ARCH=x86-64  
  4. Run this test: python test_engine.py

  INTERPRETING RESULTS:
  - Higher NPS = faster search (good!)
  - Lower nodes at same depth = more efficient pruning (good!)
  - Score changes might indicate evaluation changes

  TIPS:
  - Test one change at a time
  - Run multiple times to account for variance
  - For serious testing, use Fishtest
""")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("  STOCKFISH PERFORMANCE TESTER")
    print("="*60)
    
    # Check if engines exist
    baseline_exists = os.path.exists(BASELINE_ENGINE)
    modified_exists = os.path.exists(MODIFIED_ENGINE)
    
    if not baseline_exists:
        print(f"\n⚠️  Baseline engine not found: {BASELINE_ENGINE}")
        print("   Please copy your original stockfish.exe to stockfish_baseline.exe")
    
    if not modified_exists:
        print(f"\n⚠️  Modified engine not found: {MODIFIED_ENGINE}")
        print("   Please build stockfish.exe first")
    
    if not baseline_exists or not modified_exists:
        run_summary()
        return
    
    # Parse arguments
    args = sys.argv[1:]
    run_all = len(args) == 0
    
    if run_all or "--bench" in args:
        run_bench_test(BASELINE_ENGINE, MODIFIED_ENGINE)
    
    if run_all or "--positions" in args:
        run_position_test(BASELINE_ENGINE, MODIFIED_ENGINE)
    
    print("\n" + "="*60)
    print("  TESTING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
