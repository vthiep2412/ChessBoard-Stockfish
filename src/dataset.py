import chess
import numpy as np
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, games, board_converter=None):
        """
        games: List of tuples (board_state, policy_target, value_target)
               or a pgn file path (to be implemented)
        """
        self.games = games

    @staticmethod
    def load_from_pgn(pgn_file, max_games=1000):
        """
        Parses a PGN file and extracts training examples.
        Returns a list of (board_tensor, policy_target, value_target).
        """
        import chess.pgn
        from tqdm import tqdm
        
        examples = []
        with open(pgn_file) as f:
            for _ in tqdm(range(max_games), desc="Parsing Games"):
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    game_value = 1.0
                elif result == "0-1":
                    game_value = -1.0
                elif result == "1/2-1/2":
                    game_value = 0.0
                else:
                    continue # Skip unfinished games
                
                board = game.board()
                for move in game.mainline_moves():
                    # Input: Board state
                    board_tensor = encode_board(board)
                    
                    # Target Policy: The move that was actually played
                    # In a real comprehensive project, we'd use search probabilities (MCTS) 
                    # from self-play as targets. For supervised PGN learning, 
                    # we train the network to predict the GM's move (Behavior Cloning).
                    policy_idx = decode_move_to_policy_index(move)
                    
                    # Target Value: The final result of the game (from current player perspective)
                    # If White won (1.0) and it's White's turn, target is 1.0
                    # If White won (1.0) and it's Black's turn, target is -1.0
                    current_turn_value = game_value if board.turn == chess.WHITE else -game_value
                    
                    examples.append((board_tensor, policy_idx, torch.tensor(current_turn_value, dtype=torch.float32)))
                    
                    board.push(move)
                    
        return ChessDataset(examples)

    @staticmethod
    def load_from_self_play(data_dir, max_files=100):
        """
        Loads self-play data from pickle files in a directory.
        """
        import pickle
        import os
        import glob
        
        examples = []
        files = glob.glob(os.path.join(data_dir, "*.pkl"))
        # specific sort to get latest? or random?
        files.sort(key=os.path.getmtime, reverse=True)
        
        count = 0
        for fpath in files:
            if count >= max_files: break
            try:
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                    # data is list of (state, policy, value)
                    # state is numpy, convert to tensor
                    # policy is numpy, convert to tensor
                    # value is float, convert to tensor
                    for (s, p, v) in data:
                        s_t = torch.tensor(s, dtype=torch.float32)
                        p_t = torch.tensor(p, dtype=torch.float32) 
                        v_t = torch.tensor(v, dtype=torch.float32)
                        examples.append((s_t, p_t, v_t))
                count += 1
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                
        print(f"Loaded {len(examples)} examples from {count} self-play files.")
        return ChessDataset(examples)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        board_tensor, policy, value = self.games[idx]
        return board_tensor, policy, value

def encode_board(board: chess.Board):
    """
    Encodes a python-chess Board object into a tensor representation.
    Shape: (18, 8, 8)
    Planes 0-11: Pieces (P, N, B, R, Q, K) for White then Black
    Plane 12: Turn (0 for White, 1 for Black usually, or all 1s if Black to move?) 
              *AlphaZero standard: All 1s if Player A to move, 0s if Player B.*
              Let's use: All 1s if it's the current player's turn (relative viewpoint)
              BUT for simplicity in this engine, let's stick to absolute board state + color plane.
              Let's do: 12 planes (White P..K, Black P..K).
              Plane 12: 1.0 if White to move, 0.0 if Black.
              Plane 13-16: Castling rights (WK, WQ, BK, BQ)
              Plane 17: Repetition count or simple 0 (placeholder for now)
    """
    # 18 planes
    # 0-5: White Pieces [P, N, B, R, Q, K]
    # 6-11: Black Pieces [P, N, B, R, Q, K]
    # 12: Turn (1 for White, 0 for Black)
    # 13: White King Castling
    # 14: White Queen Castling
    # 15: Black King Castling
    # 16: Black Queen Castling
    # 17: Move count / 50 rule (normalized) ?? Or just En Passant square?
    # Let's use Plane 17 for En Passant target existence.
    
    state = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Piece positions
    # chess.PAWN=1 ... chess.KING=6
    for piece_type in range(1, 7):
        # White pieces
        for square in board.pieces(piece_type, chess.WHITE):
            row = chess.square_rank(square)
            col = chess.square_file(square)
            state[piece_type - 1, row, col] = 1.0
        
        # Black pieces
        for square in board.pieces(piece_type, chess.BLACK):
            row = chess.square_rank(square)
            col = chess.square_file(square)
            state[piece_type + 5, row, col] = 1.0
            
    # Turn
    if board.turn == chess.WHITE:
        state[12, :, :] = 1.0
        
    # Castling Rights
    if board.has_kingside_castling_rights(chess.WHITE):
        state[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        state[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        state[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        state[16, :, :] = 1.0
        
    # En Passant
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        state[17, row, col] = 1.0
        
    return torch.tensor(state)

def decode_move_to_policy_index(move: chess.Move):
    """
    Maps a chess.Move to an index [0, 4671] (simplified or full).
    For a simplified version, we can just use picking the best move from the list directly
    without a fixed policy vector output if we want to save complexity.
    However, AlphaZero uses a fixed policy head output (73*8*8 = 4672).
    
    Let's implement a simplified flattened 64*64 = 4096 approach for "from-to" moves
    to handle the vast majority of moves easily for this project scope.
    Structure: from_square (0-63) * 64 + to_square (0-63).
    Indicies 0 to 4095. 
    Promotions: This simplified 64*64 schema misses promotion types (Q,R,B,N).
    We will assume Queen promotion for this simple mapping or add offsets.
    
    Let's stick to the 4096 (64x64) for simplicity.
    Promotions will default to Queen in this encoding (lossy).
    """
    from_sq = move.from_square
    to_sq = move.to_square
    return from_sq * 64 + to_sq

def decode_policy_index_to_move(idx: int, board: chess.Board):
    """
    Converts policy index back to move.
    """
    from_sq = idx // 64
    to_sq = idx % 64
    
    # Create move - handle promotions roughly (always Queen for now if pawn on 7th/2nd rank)
    promotion = None
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or \
           (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0):
            promotion = chess.QUEEN
            
    move = chess.Move(from_sq, to_sq, promotion=promotion)
    return move
