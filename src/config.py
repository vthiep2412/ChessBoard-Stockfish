import torch

class Config:
    # Model
    NUM_RESIDUAL_BLOCKS = 5  # Start small for faster debugging/training
    NUM_FILTERS = 128
    
    # Input Data
    # 12 piece planes + 1 turn + 4 castling + 1 en-passant = 18 planes
    INPUT_SHAPE = (18, 8, 8) 
    
    # MCTS
    NUM_MCTS_SIMULATIONS = 400
    C_PUCT = 1.0
    DIRICHLET_ALPHA = 0.3 # Exploration noise (0.3 for chess)
    
    # Increase Batch Size for GPU speed (RTX 4050 can handle this easily)
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
