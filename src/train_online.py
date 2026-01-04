import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob

# Allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import ChessResNet
from src.dataset import ChessDataset

def train_online(last_n_games=50):
    """
    Fine-tunes the model on the most recent Arena games.
    If last_n_games is None, uses ALL available games.
    """
    device = Config.DEVICE
    print(f"Online Training on {device}...")
    
    # 1. Find recent PGNs
    arena_dir = os.path.join(os.path.dirname(__file__), "..", "data", "arena_games")
    if not os.path.exists(arena_dir):
        print("No arena games found.")
        return

    files = glob.glob(os.path.join(arena_dir, "*.pgn"))
    # Sort by time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Take top N if specified
    if last_n_games:
        recent_files = files[:last_n_games]
    else:
        # Safety fallback: If somehow None is passed but we want speed, cap it?
        # No, if None, user wants ALL. But defaults to 50 now.
        recent_files = files # All games
        
    if not recent_files:
        print("No PGN files found.")
        return
        
    print(f"Loading {len(recent_files)} recent games for fine-tuning...")
    
    # 2. Load Data
    # specific load method for multiple files? 
    # Dataset.load_from_pgn takes a single file path usually.
    # We'll assume we can concatenate or just iterate?
    # Actually, let's just make a temporary combined PGN or load one by one.
    # Loading one by one is inefficient for batch norm etc.
    # Dataset currently doesn't support list of files.
    # Let's simple-hack: Read all content into one string or temp file.
    
    combined_pgn_path = os.path.join(arena_dir, "temp_online_batch.pgn")
    with open(combined_pgn_path, "w") as outfile:
        for fname in recent_files:
            with open(fname) as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")
                
    dataset = ChessDataset.load_from_pgn(combined_pgn_path)
    
    if len(dataset) == 0:
        print("No valid positions extracted.")
        return

    # 3. Load Model
    model = ChessResNet().to(device)
    model_path = "model.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded current model.")
        except:
            print("Could not load model, starting fresh (Not recommended for online).")
    
    model.train()
    
    # 4. Training Setup - Low LR for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower LR than main training
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Small batch size for online update
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Train 1 Epoch
    total_loss = 0
    # Enable AMP
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Training on {len(dataset)} positions...")
    
    for boards, policy_targets, value_targets in dataloader:
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device).float()
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            pred_policy, pred_value = model(boards)
            
            # Arena data is PGN, so targets are indices (Supervised style)
            # But we might want to blend with soft targets later. For now, hard targets.
            if policy_targets.dim() == 1:
                loss_p = policy_criterion(pred_policy, policy_targets)
            else:
                 # Should not happen for PGN data unless we change dataset
                log_probs = torch.log_softmax(pred_policy, dim=1)
                loss_p = nn.KLDivLoss(reduction='batchmean')(log_probs, policy_targets)
                
            # Value Loss - Always calculate
            loss_v = value_criterion(pred_value.view(-1), value_targets)
            loss = loss_p + loss_v
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    print(f"Online Update Complete. Avg Loss: {total_loss/len(dataloader):.4f}")
    
    # 5. Save
    torch.save(model.state_dict(), model_path)
    print("Model updated.")
    
    # Cleanup
    if os.path.exists(combined_pgn_path):
        os.remove(combined_pgn_path)

if __name__ == "__main__":
    train_online()
