import sys
import os
import datetime

# Add current directory to path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy imports
# from src.config import Config
# from src.model import ChessResNet
# from src.mcts import MCTS
# from src.dataset import decode_policy_index_to_move

def main():
    # Setup Logging
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uci_debug.log")
    def log(msg):
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} - {msg}\n")
    
    log("UCI Script Started.")
    
    # Global state variables
    model = None
    mcts = None
    device = None
    chess_module = None
    
    # We need basic chess for UCI identity, but let's lazy load that too if possible?
    # Actually python-chess is fast.
    import chess
    board = chess.Board()

    def log(msg):
        pid = os.getpid()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"uci_debug_{pid}.log")
        with open(path, "a") as f:
            f.write(f"{datetime.datetime.now()} - {msg}\n")
            
    log("UCI Loop Starting...")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                log("EOF received. Exiting.")
                break
            line = line.strip()
            log(f"CMD RECV: {line}")
            
            if line == "uci":
                print("id name AntigravityZero")
                print("id author Antigravity")
                print("uciok")
                sys.stdout.flush()
                log("SENT: uciok")
            
            elif line == "isready":
                log("PROCESSING: isready")
                if model is None:
                    # Lazy Load
                    log("Lazy Loading Modules...")
                    import torch
                    from src.config import Config
                    from src.model import ChessResNet
                    from src.mcts import MCTS
                    
                    device = Config.DEVICE
                    log(f"Device: {device}")
                    model = ChessResNet().to(device)
                    
                    # Load weights
                    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pth")
                    log(f"Loading weights from {model_path}")
                    if os.path.exists(model_path):
                        try:
                            model.load_state_dict(torch.load(model_path, map_location=device))
                            log("Weights loaded successfully.")
                        except Exception as e:
                            log(f"Weight load failed: {e}")
                            pass
                    model.eval()
                    mcts = MCTS(model, device=device)
                    log("MCTS Initialized.")
                    
                print("readyok")
                sys.stdout.flush()
                log("SENT: readyok")
            
            elif line == "ucinewgame":
                log("PROCESSING: ucinewgame")
                # MCTS tree is automatically cleared when run() creates a new root node.
                # Model stays loaded - DO NOT reload PyTorch!
                board = chess.Board()
                log("New Game Ready (Instant).")
            
            elif line.startswith("position"):
                log(f"PROCESSING: position {line[:20]}...")
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
                        
            elif line.startswith("go"):
                log("PROCESSING: go")
                if mcts is None:
                     # Should not happen if protocol followed, but safety fallback
                     print("bestmove 0000")
                     sys.stdout.flush()
                     continue
                     
                root = mcts.run(board)
                
                best_move = None
                max_visits = -1
                
                for move, node in root.children.items():
                    if node.visit_count > max_visits:
                        max_visits = node.visit_count
                        best_move = move
                
                if best_move:
                    print(f"bestmove {best_move.uci()}")
                    log(f"SENT: bestmove {best_move.uci()}")
                else:
                    if not board.is_game_over():
                        print(f"bestmove {list(board.legal_moves)[0].uci()}")
                    else:
                        print("bestmove 0000")
                        
                sys.stdout.flush()
                
            elif line == "quit":
                log("Quit received.")
                break
                
        except Exception as e:
            msg = f"LOOP ERROR: {e}"
            # Log to file
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uci_debug.log")
            with open(path, "a") as f:
                 f.write(f"{datetime.datetime.now()} - {msg}\n")
                 import traceback
                 traceback.print_exc(file=f)
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("uci_critical.log", "a") as f:
            f.write(f"Critical Error: {e}\n")
