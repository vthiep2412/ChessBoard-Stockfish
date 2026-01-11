#!/usr/bin/env python3
"""
AntigravityZero Classical - Pure Python, No Neural Network
Uses only heuristics.py for evaluation and search.
Instant startup - no PyTorch loading!
"""
import sys
import os
import datetime

# Add parent directory to path BEFORE importing chess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
from src.heuristics import Heuristics

def main():
    # Setup Logging
    pid = os.getpid()
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"classical_debug_{pid}.log")
    
    def log(msg):
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} - {msg}\n")
    
    log("Classical UCI Engine Started (No NN).")
    board = chess.Board()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                log("EOF received. Exiting.")
                break
            line = line.strip()
            log(f"CMD RECV: {line}")
            
            if line == "uci":
                print("id name AntigravityClassical")
                print("id author Antigravity")
                print("option name Depth type spin default 10 min 1 max 20")
                print("uciok")
                sys.stdout.flush()
                log("SENT: uciok")
            
            elif line == "isready":
                # Instant ready - no heavy loading!
                print("readyok")
                sys.stdout.flush()
                log("SENT: readyok")
            
            elif line == "ucinewgame":
                board = chess.Board()
                log("New Game Ready (Instant).")
            
            elif line.startswith("position"):
                parts = line.split()
                moves_idx = -1
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                
                if "startpos" in parts:
                    board = chess.Board()
                elif "fen" in parts:
                    if moves_idx != -1:
                        fen_parts = parts[parts.index("fen")+1 : moves_idx]
                    else:
                        fen_parts = parts[parts.index("fen")+1 :]
                    fen = " ".join(fen_parts)
                    board = chess.Board(fen)
                
                if moves_idx != -1:
                    moves = parts[moves_idx+1:]
                    for move_uci in moves:
                        board.push(chess.Move.from_uci(move_uci))
                        
                log(f"Position set: {board.fen()}")
                        
            elif line.startswith("go"):
                log("PROCESSING: go")
                
                # Parse depth (default 10)
                depth = 10
                parts = line.split()
                if "depth" in parts:
                    try:
                        depth = int(parts[parts.index("depth") + 1])
                    except:
                        pass
                
                # Cap at 4 for Python (6+ is too slow!)
                depth = min(depth, 4)
                
                # Use heuristics.py negamax search
                try:
                    color_sign = 1 if board.turn == chess.WHITE else -1
                    score, best_move = Heuristics.negamax(
                        board, 
                        depth, 
                        alpha=-float('inf'), 
                        beta=float('inf'), 
                        color_sign=color_sign
                    )
                    log(f"Search returned: {best_move} (score: {score})")
                    
                    if best_move:
                        print(f"bestmove {best_move.uci()}")
                    else:
                        # Fallback to first legal move
                        moves = list(board.legal_moves)
                        if moves:
                            print(f"bestmove {moves[0].uci()}")
                        else:
                            print("bestmove 0000")
                except Exception as e:
                    log(f"Search error: {e}")
                    import traceback
                    log(traceback.format_exc())
                    # Fallback
                    moves = list(board.legal_moves)
                    if moves:
                        print(f"bestmove {moves[0].uci()}")
                    else:
                        print("bestmove 0000")
                        
                sys.stdout.flush()
                log(f"SENT: bestmove")
                
            elif line == "quit":
                log("Quit received.")
                break
                
        except Exception as e:
            log(f"LOOP ERROR: {e}")
            import traceback
            log(traceback.format_exc())
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("classical_critical.log", "a") as f:
            f.write(f"Critical Error: {e}\n")
            traceback.print_exc(file=f)
