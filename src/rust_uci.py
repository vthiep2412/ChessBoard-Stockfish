#!/usr/bin/env python3
"""
Pure Rust UCI Engine
No PyTorch, no neural network - just fast alpha-beta search
"""
import sys
import os
import datetime
import chess

# Import the Rust engine
import rust_engine

def main():
    # Change to Chess directory so book paths work
    chess_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(chess_dir)
    
    # Setup Logging
    pid = os.getpid()
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rust_uci_debug_{pid}.log")
    
    def log(msg):
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} - {msg}\n")
    
    log(f"Rust UCI Engine Started. Working dir: {os.getcwd()}")
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
                print("id name AntigravityRust")
                print("id author Antigravity")
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
                # Clear transposition table for fresh game!
                try:
                    rust_engine.clear_tt()
                    log("New Game Ready - TT Cleared.")
                except Exception as e:
                    log(f"New Game Ready (clear_tt failed: {e}).")
            
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
                
                # Load config
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine_config.json")
                depth = 22 # default fallback
                override = False
                aggressiveness = 5  # 1-10, higher = more pruning = faster
                use_parallel = True  # Enable multi-core search
                
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            if "default_depth" in config:
                                depth = int(config["default_depth"])
                            if "override_gui_depth" in config:
                                override = config["override_gui_depth"]
                            if "aggressiveness" in config:
                                aggressiveness = int(config["aggressiveness"])
                            if "use_parallel" in config:
                                use_parallel = config["use_parallel"]
                    except Exception as e:
                        log(f"Config load error: {e}")

                parts = line.split()
                if "depth" in parts and not override:
                    try:
                        depth = int(parts[parts.index("depth") + 1])
                    except Exception as e:
                        log(f"Depth parse error: {e}")
                
                # Call Rust engine!
                try:
                    log(f"Searching depth {depth}, aggr {aggressiveness}, parallel={use_parallel}...")
                    best_move = rust_engine.get_best_move(board.fen(), depth, aggressiveness, use_parallel)
                    log(f"Rust returned: {best_move}")
                    
                    if best_move and best_move != "0000":
                        print(f"bestmove {best_move}")
                    else:
                        # Fallback to first legal move
                        moves = list(board.legal_moves)
                        if moves:
                            print(f"bestmove {moves[0].uci()}")
                        else:
                            print("bestmove 0000")
                except Exception as e:
                    log(f"Rust error: {e}")
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
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("rust_uci_critical.log", "a") as f:
            f.write(f"Critical Error: {e}\n")
