import sys
import os
import time

# Allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_play import run_self_play
from src.train import train_model
from src.dataset import ChessDataset

def rl_loop():
    iteration = 0
    data_dir = "data/self_play"
    
    print("Starting Reinforcement Learning Loop...")
    
    while True:
        iteration += 1
        print(f"\n=== RL Iteration {iteration} ===")
        
        # 1. Self Play
        print(">> Phase 1: Self-Play (Generating Games)...")
        # Generate 10 games per iteration (smaller batch for faster feedback loop)
        run_self_play(num_games=10, save_dir=data_dir)
        
        # 2. Training
        print(">> Phase 2: Training...")
        # Load dataset (all generated files or last N files)
        # Using max_files=500 to keep a rolling buffer of recent games (replay buffer)
        dataset = ChessDataset.load_from_self_play(data_dir, max_files=500)
        
        if len(dataset) > 0:
            print(f"Training on {len(dataset)} positions...")
            # Train for 1 epoch on this buffer (or more?)
            # Usually 1-5 epochs on the new data is good.
            # We reuse train_model which saves to 'model.pth'
            train_model(dataset, save_path="model.pth")
        else:
            print("No data found, skipping training.")
            
        print(f"Iteration {iteration} complete.")
        # Optional: Eval against stockfish here?

if __name__ == "__main__":
    rl_loop()
