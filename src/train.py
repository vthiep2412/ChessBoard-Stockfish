import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Allow imports from src when running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import ChessResNet
from src.dataset import ChessDataset

def train_model(dataset: ChessDataset, save_path="model.pth"):
    device = Config.DEVICE
    print(f"Training on device: {device}")
    
    model = ChessResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Loss functions
    # Policy: Cross Entropy (using logits)
    # Value: MSE
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # pin_memory=True speeds up transfer to GPU
    # num_workers=2 with persistence: High throughput, one-time startup cost.
    # prefetch_factor=2: Buffers data to eliminate delays.
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    # Enable cuDNN benchmark for speed
    torch.backends.cudnn.benchmark = True
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        for batch_idx, (boards, policy_targets, value_targets) in enumerate(dataloader):
            boards = boards.to(device)
            policy_targets = policy_targets.to(device) # Shape: (B, 4096) or indices?
            # If policy_targets are probabilities, use KLDiv or similar. 
            # If indices of best move, use CrossEntropy.
            # Support for both Index targets (Supervised) and Soft Probabilities (RL/AlphaZero)
            # policy_targets shape: (B) for indices, or (B, 4096) for probabilities
            value_targets = value_targets.to(device).float()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred_policy, pred_value = model(boards)

                if policy_targets.dim() == 1:
                    # Supervised PGN (Indices)
                    loss_p = policy_criterion(pred_policy, policy_targets)
                else:
                    # RL Self-Play (Probabilities)
                    # pred_policy is logits. Target is probs. Use KLDiv or Soft CrossEntropy.
                    # KLDiv in pytorch expects log_softmax input.
                    log_probs = torch.log_softmax(pred_policy, dim=1)
                    loss_p = nn.KLDivLoss(reduction='batchmean')(log_probs, policy_targets)
                
                # Value Loss
                # prevent squeeze() from creating scalar if batch=1, use view(-1) to ensure 1D
                loss_v = value_criterion(pred_value.view(-1), value_targets)
                
                loss = loss_p + loss_v
            
            # Scaled Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} Complete. Average Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import os
    
    # Path to the downloaded PGN
    pgn_path = os.path.join(os.path.dirname(__file__), "..", "data", "grandmaster_games.pgn")
    
    if os.path.exists(pgn_path):
        print(f"Loading games from {pgn_path}...")
        # Load up to 10000 games if available
        dataset = ChessDataset.load_from_pgn(pgn_path, max_games=10000)
        
        if len(dataset) > 0:
            print(f"Starting training on {len(dataset)} moves/positions...")
            train_model(dataset, save_path="model.pth")
        else:
            print("No valid games found in PGN.")
    else:
        print(f"PGN file not found at {pgn_path}")
        print("Please run 'python src/download_data.py' first.")
