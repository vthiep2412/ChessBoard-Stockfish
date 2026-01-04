import torch
import numpy as np
import chess
import os
import pickle
import time
from tqdm import tqdm
import sys

# Allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import ChessResNet
from src.mcts import MCTS
from src.dataset import encode_board, decode_move_to_policy_index

def get_mcts_policy(root_node):
    """
    Converts MCTS root children visit counts to a policy probability vector (4096).
    """
    policy = np.zeros(4096, dtype=np.float32)
    sum_visits = 0
    
    for move, child in root_node.children.items():
        idx = decode_move_to_policy_index(move)
        if idx < 4096:
            policy[idx] = child.visit_count
            sum_visits += child.visit_count
            
    if sum_visits > 0:
        policy /= sum_visits
    else:
        # Should not happen typically
        policy[:] = 1.0 / 4096.0
        
    return policy

def run_self_play(num_games=100, save_dir="data/self_play"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    device = Config.DEVICE
    model = ChessResNet().to(device)
    
    # Load best model if exists
    if os.path.exists("model.pth"):
        try:
            model.load_state_dict(torch.load("model.pth", map_location=device))
            print("Loaded existing model for self-play.")
        except:
            print("Could not load model.pth, starting from scratch/random.")
    else:
        print("No model.pth found. Using random weights.")
    
    model.eval()
    mcts = MCTS(model, device=device)
    
    games_data = [] # List of (state, policy, value)
    
    print(f"Starting {num_games} self-play games on {device}...")
    
    for game_idx in range(num_games):
        board = chess.Board()
        game_history = [] # Stores (encoded_state, policy, player_color)
        
        move_count = 0
        while not board.is_game_over(claim_draw=True):
            move_count += 1
            
            # Run MCTS
            # Reduce simulations for speed during self-play generation? 
            # Or keep high for quality? Let's stick to Config default or slightly lower.
            # Using 200 for speed vs 400 default.
            root = mcts.run(board, num_simulations=200)
            
            # Get Policy from visit counts
            policy = get_mcts_policy(root)
            
            # Encode current state
            state_tensor = encode_board(board).cpu().numpy() # [18, 8, 8]
            
            # Store
            game_history.append((state_tensor, policy, board.turn))
            
            # Select Move
            # Temperature: High early in game (exploration), Low later.
            if move_count < 30:
                # Sample from distribution
                action_idx = np.random.choice(4096, p=policy)
            else:
                # Greedy (most visited)
                action_idx = np.argmax(policy)
            
            # Find the move object corresponding to action_idx
            # We need to map back carefully or just pick child with max visits
            # Since decode is lossy, safer to pick from root.children
            
            if move_count < 30:
                # We sampled an index, finding matching move might be tricky if multiple moves map to same index (promotions).
                # But our decode assumes Queens. 
                # Better approach: Sanple from legal moves using the policy probabilities we just calculated.
                 # Reconstruct legal moves list to match policy distribution construction
                legal_moves = []
                probs = []
                for move, child in root.children.items():
                    legal_moves.append(move)
                    probs.append(child.visit_count)
                
                probs = np.array(probs)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    move = np.random.choice(legal_moves, p=probs)
                else:
                    move = np.random.choice(list(board.legal_moves))
            else:
                # Argmax
                best_visits = -1
                move = None
                for m, child in root.children.items():
                    if child.visit_count > best_visits:
                        best_visits = child.visit_count
                        move = m
            
            board.push(move)
            
            # Hard limit for game length to prevent infinite loops
            if move_count > 200:
                break
        
        # Game Over
        res = board.result(claim_draw=True)
        if res == "1-0":
            z = 1.0
        elif res == "0-1":
            z = -1.0
        else:
            z = 0.0
            
        print(f"Game {game_idx+1}/{num_games} finished: {res} ({move_count} moves)")
        
        # Assign values to history
        # If result is White Win (1.0):
        # State where it was White's turn -> Target = 1.0
        # State where it was Black's turn -> Target = -1.0
        for state, pi, turn in game_history:
            if turn == chess.WHITE:
                val = z
            else:
                val = -z
            games_data.append((state, pi, val))
            
        # Save periodically
        if (game_idx + 1) % 10 == 0:
            filename = os.path.join(save_dir, f"self_play_{int(time.time())}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(games_data, f)
            print(f"Saved {len(games_data)} positions to {filename}")
            games_data = [] # Clear buffer

    # Final save
    if games_data:
        filename = os.path.join(save_dir, f"self_play_{int(time.time())}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(games_data, f)

if __name__ == "__main__":
    # Continuous loop
    while True:
        run_self_play(num_games=10)
        # In a real loop, we would trigger training here or separate process
