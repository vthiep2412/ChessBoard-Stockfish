import math
import torch
import torch.nn.functional as F
import numpy as np
import chess
from .config import Config
from .dataset import encode_board, decode_move_to_policy_index, decode_policy_index_to_move
from .heuristics import Heuristics

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}  # Map from move (or move_idx) to Node
        self.state = None # Optional: store board state? Memory heavy for large trees.

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else Config.DEVICE
        self.model.eval()

    def run(self, board, num_simulations=None):
        # SPEED OVERRIDE: 50 Simulations Max for rapid play
        if num_simulations is None:
            num_simulations = 50 # Was Config.NUM_MCTS_SIMULATIONS (800)
        
        # Root node
        root = Node(0)
        
        # We need to expand the root first to get legal moves
        self.expand_root(root, board, noise=(Config.DIRICHLET_ALPHA is not None))
        
        for _ in range(num_simulations):
            node = root
            scratch_board = board.copy()
            search_path = [node]
            
            # 1. Select
            while node.children:
                move, node = self.select_child(node)
                scratch_board.push(move)
                search_path.append(node)
                
            # 2. Expand and Evaluate
            value = self.evaluate(node, scratch_board)
            
            # 3. Backpropagate
            self.backpropagate(search_path, value, scratch_board.turn)
            
        return root

    def expand_root(self, root, board, noise=False):
        """
        Special expansion for root to ensure we only consider legal moves
        and add noise for exploration (if training).
        """
        value = self.evaluate(root, board)
        
        # Add Dirichlet Noise
        if noise:
            # Alpha=0.3 is standard for Chess
            alpha = 0.3
            epsilon = 0.25 # 25% noise, 75% policy
            
            moves = list(root.children.keys())
            noise_v = np.random.dirichlet([alpha] * len(moves))
            
            for i, move in enumerate(moves):
                node = root.children[move]
                node.prior = (1 - epsilon) * node.prior + epsilon * noise_v[i]
        
        return value

    def evaluate(self, node, board):
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                # If White wins, and it's White's turn -> +1
                # If White wins, and it's Black's turn -> -1 (Bad for current player)
                # But wait, MCTS 'value' is usually for the player who JUST moved (parent).
                # No, standard is: Value of state S is for player P whose turn it is in S.
                # If S is terminal/WhiteWin. P(White) sees +1. P(Black) sees -1.
                # board.result "1-0" means absolute White win.
                return 1 if board.turn == chess.WHITE else -1
            elif result == "0-1":
                return -1 if board.turn == chess.WHITE else 1
            else:
                return 0

        # 1. Neural Net Evaluation (Relative)
        x = encode_board(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(x)
        
        nn_value = value.item()
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        
        # 2. Heuristic Evaluation (Absolute -> Relative)
        # Reverted to Static Eval for speed. Q-Search was too slow in Python.
        # But we KEEP the threat penalty logic in evaluate()!
        heur_score = Heuristics.evaluate(board) 
        if board.turn == chess.BLACK:
            heur_score = -heur_score # Convert to "Good for Black"
            
        # 3. Hybrid Mix
        # Dynamic Alpha:
        # If the NN is very confident (sees a win/loss), we trust it more (Alpha reduces).
        # If the NN is unsure (-0.5 to 0.5), we trust the Classical Rules (Alpha high).
        
        if abs(nn_value) > 0.8:
            # "I see a mate/win!" - Trust the Brain.
            alpha = 0.3
        elif abs(nn_value) > 0.5:
             # "I have a strong feeling."
             alpha = 0.5
        else:
            # "I don't know, let's play safe."
            alpha = 0.8

        hybrid_value = (1 - alpha) * nn_value + alpha * heur_score
        
        # Expand children (only legal moves)
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            idx = decode_move_to_policy_index(move)
            if idx < len(policy):
                prob = policy[idx]
                
                # 4. Tactical Safety Bias (Anti-Blunder)
                safety_bias = Heuristics.get_move_safety_bias(board, move)
                # safety_bias = 0
                
                if safety_bias < -500:
                    # Hanging Queen or Rook? Almost forbidden.
                    # Only searched if NN is 100% sure it's a mate.
                    prob *= 0.001 
                elif safety_bias < -200:
                    # Hanging Minor Piece? Severe penalty.
                    prob *= 0.01 # Was 0.05. Now 1%!
                elif safety_bias < 0:
                    # Hanging Pawn? Discourage.
                    prob *= 0.1 # Was 0.2. Now 10%.
                elif safety_bias > 0:
                     prob *= 1.5 # Boost good captures
                
                node.children[move] = Node(prior=prob)
            else:
                pass

        return hybrid_value

    def select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            ucb_score = self.ucb_score(node, child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def ucb_score(self, parent, child):
        prior_score = Config.C_PUCT * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        # Q-value usually needs to be normalized or aligned with player perspective
        # Here we simple average
        if child.visit_count > 0:
             value_score = -child.value() # Invert because child's value is from opponent's perspective (after making move)
        else:
            value_score = 0
            
        return value_score + prior_score

    def backpropagate(self, search_path, value, turn_at_leaf):
        """
        Update nodes with value. 
        Value is from perspective of 'turn_at_leaf'.
        If the node's turn (who JUST moved to get there) matches value perspective, add value.
        Else subtract unique perspective.
        
        Actually, simpler standard convention:
        Value is +1 (White wins).
        If current node is White's turn to move, a high value child means good for White.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value # Toggle perspective for parent
