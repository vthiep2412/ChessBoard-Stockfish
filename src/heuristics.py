import chess

class Heuristics:
    # PeSTO Piece-Square Tables (Tapered Evaluation)
    # Source: https://www.chessprogramming.org/PeSTO
    # Scores are in centipawns (cp)
    
    # Material Values (Middlegame, Endgame) - roughly
    MATERIAL_MG = {chess.PAWN: 82, chess.KNIGHT: 337, chess.BISHOP: 365, chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0}
    MATERIAL_EG = {chess.PAWN: 94, chess.KNIGHT: 281, chess.BISHOP: 297, chess.ROOK: 512, chess.QUEEN: 936, chess.KING: 0}

    # Tables (Flip for black)
    # We will use simplified 64-length arrays for readability
    
    # Pawns
    MG_PAWN = [
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    ]
    EG_PAWN = [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ]

    # Knights
    MG_KNIGHT = [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ]
    EG_KNIGHT = [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -32, -18, -84,   -9,
    ]

    # Bishops
    MG_BISHOP = [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ]
    EG_BISHOP = [
        -14, -21, -11,  -8,  -7,  -9, -17, -24,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -3,   9,  12,   9,  14,  10,   3,   2,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    ]

    # Rooks
    MG_ROOK = [
         32,  42,  32,  51,  63,   9,  31,  43,
         27,  32,  58,  62,  80,  67,  26,  44,
         -5,  19,  26,  36,  17,  45,  61,  16,
        -24, -11,   7,  26,  24,  35,  -8, -20,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -19, -13,   1,  17,  16,   7, -37, -26,
    ]
    EG_ROOK = [
         13,  10,  18,  15,  12,  12,   8,   5,
         11,  13,  13,  11,  12,  12,  11,  11,
          7,   7,   7,   5,   4,  -3,  -5,  -6,
          4,   3,  13,   1,   2,   1,  -1,   2,
          3,   5,   8,   4,  -5,  -6,  -8, -11,
         -4,   0,  -5,  -1,  -7, -12,  -8, -16,
         -6,  -6,   0,   2,  -9,  -9, -11,  -3,
         -9,   2,   3,  -1,  -5, -13,   4, -20,
    ]

    # Queens
    MG_QUEEN = [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26, -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9, -19, -30, -15, -13, -22,
    ]
    EG_QUEEN = [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ]

    # Kings
    MG_KING = [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ]
    EG_KING = [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ]

    # Map piece type to (MG_TABLE, EG_TABLE, MG_VAL, EG_VAL)
    TABLES = {
        chess.PAWN:   (MG_PAWN, EG_PAWN, MATERIAL_MG[chess.PAWN], MATERIAL_EG[chess.PAWN]),
        chess.KNIGHT: (MG_KNIGHT, EG_KNIGHT, MATERIAL_MG[chess.KNIGHT], MATERIAL_EG[chess.KNIGHT]),
        chess.BISHOP: (MG_BISHOP, EG_BISHOP, MATERIAL_MG[chess.BISHOP], MATERIAL_EG[chess.BISHOP]),
        chess.ROOK:   (MG_ROOK, EG_ROOK, MATERIAL_MG[chess.ROOK], MATERIAL_EG[chess.ROOK]),
        chess.QUEEN:  (MG_QUEEN, EG_QUEEN, MATERIAL_MG[chess.QUEEN], MATERIAL_EG[chess.QUEEN]),
        chess.KING:   (MG_KING, EG_KING, MATERIAL_MG[chess.KING], MATERIAL_EG[chess.KING]),
    }
    
    # Phase calculation
    # Max phase = 4*Knight + 4*Bishop + 4*Rook + 2*Queen
    # Limits for tapered eval
    # Assign phase weights: N=1, B=1, R=2, Q=4
    # Total = 4*1 + 4*1 + 4*2 + 2*4 = 24
    PHASE_WEIGHTS = {
        chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4
    }
    TOTAL_PHASE = 24

    @staticmethod
    def get_eval_score(board):
        """
        Returns tapered evaluation score (centipawns) from perspective of White.
        """
        mg_score = 0
        eg_score = 0
        phase = Heuristics.TOTAL_PHASE

        # Iterate all pieces
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                pt = piece.piece_type
                color = piece.color
                
                mg_table, eg_table, mg_val, eg_val = Heuristics.TABLES[pt]
                
                # PST arrays are often 0-63 a1-h8. 
                # White: index = sq. 
                # Black: index = sq with rank flipped (mirror).
                if color == chess.WHITE:
                    idx = chess.square_mirror(sq) # Wait, standard usually stores white from top?
                    # PeSTO table usually: index 0 is A1.
                    # python-chess: 0 is A1.
                    # But visual tables usually print Rank 8 at top.
                    # Standard PeSTO printed tables are usually Rank 1 at bottom.
                    # BUT many engines flip vertically for White.
                    # Let's assume index 0 = A1 (56 in printed array if top-left).
                    # Actually, python-chess `sq` 0 is A1.
                    # If I copy-pasted standard array where A1 is bottom-left (index 56 in a 8x8 printed)
                    # Let's trust standard mirroring. 
                    # Usually: White idx = sq ^ 56 (flip vertical) IF table is White-Top.
                    # Let's run with: White uses index = `sq ^ 56` (A1->A8 mapping) if table is designed visually.
                    # PeSTO tables above are printed rank 8 first.
                    # So index 0 of array is A8.
                    # python-chess 0 is A1. So we need to map A1(0) -> A1(index 56).
                    idx = sq ^ 56 
                else:
                    # Black: mirror vertically relative to White's view?
                    # No, Black at A8 should get same score as White at A1.
                    # White A1 (sq 0) -> accesses index 56.
                    # Black A8 (sq 56) -> should access index 56.
                    # So Black idx = sq.
                    idx = sq
                
                # Add/Subtract
                if color == chess.WHITE:
                    mg_score += mg_val + mg_table[idx]
                    eg_score += eg_val + eg_table[idx]
                    if pt in Heuristics.PHASE_WEIGHTS: phase -= Heuristics.PHASE_WEIGHTS[pt]
                else:
                    mg_score -= mg_val + mg_table[idx]
                    eg_score -= eg_val + eg_table[idx]
                    if pt in Heuristics.PHASE_WEIGHTS: phase -= Heuristics.PHASE_WEIGHTS[pt]

        phase = max(0, phase)
        phase = (phase * 256 + (Heuristics.TOTAL_PHASE / 2)) / Heuristics.TOTAL_PHASE
        
        # Tapered Score
        # score = (mg * phase + eg * (256 - phase)) / 256
        eval_cp = ((mg_score * phase) + (eg_score * (256 - phase))) / 256
        return int(eval_cp)

    @staticmethod
    def get_threat_penalty(board):
        """
        Calculates penalty for pieces currently under attack.
        - Hanging pieces (Attacked > Defended, or simply Attacked & Undefended).
        - Bad trades (Attacked by weaker piece).
        Returns centipawn penalty (positive value) from White's perspective (net).
        """
        penalty = 0
        white_threats = 0
        black_threats = 0
        
        # Check all pieces
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece: continue
            
            # Is it under attack?
            attackers = board.attackers(not piece.color, sq)
            if not attackers: continue
            
            # Simple "Hanging" Check: Attacked but not Defended?
            # Or "Bad Trade": Attacked by weaker piece?
            
            val = Heuristics.MATERIAL_MG.get(piece.piece_type, 0)
            
            lowest_attacker_val = 99999
            for attacker_sq in attackers:
                a_pt = board.piece_at(attacker_sq).piece_type
                a_val = Heuristics.MATERIAL_MG.get(a_pt, 0)
                if a_val < lowest_attacker_val:
                    lowest_attacker_val = a_val
            
            # Case 1: Attacked by weaker piece
            current_penalty = 0
            if lowest_attacker_val < val:
                # We will likely lose material >= (val - attacker)
                # If defended, we lose (val - attacker).
                # If undefended, we lose val.
                # Let's assume worst case or average? 
                # If attacked by Pawn, Queen is "lost" unless it moves.
                # So penalty effectively acts as "Urgency to Move".
                current_penalty = (val - lowest_attacker_val) 
            
            # Case 2: Undefended (Hanging)
            else:
                defenders = board.attackers(piece.color, sq)
                if not defenders:
                    # Hanging piece!
                    current_penalty = val
            
            if piece.color == chess.WHITE:
                white_threats += current_penalty
            else:
                black_threats += current_penalty
                
        # Net penalty from White's perspective
        # If White has threats, score goes down.
        # If Black has threats, score goes up.
        return white_threats - black_threats

    @staticmethod
    def evaluate(board):
        """Returns score normalized to [-1, 1]."""
        cp = Heuristics.get_eval_score(board)
        
        # Apply Threat Penalty
        # This makes the evaluation "dynamic" regarding safety.
        # If a Knight is hanging, score drops by ~300.
        threat_penalty = Heuristics.get_threat_penalty(board)
        cp -= threat_penalty
        
        # Normalize: 1 pawn (100cp) ~ 0.1?
        # Let's say +/- 1000cp (10 pawns/queens) is max 1.0
        # Tanh is good for soft cap
        return max(-1.0, min(1.0, cp / 1000.0))

    @staticmethod
    def _mvv_lva_sort_key(board, move):
        """
        MVV-LVA: Most Valuable Victim - Least Valuable Attacker.
        Higher score = better capture.
        """
        if not board.is_capture(move):
            return 0
        
        victim = board.piece_at(move.to_square)
        if victim:
            victim_val = Heuristics.MATERIAL_MG.get(victim.piece_type, 0)
        elif board.is_en_passant(move):
            victim_val = Heuristics.MATERIAL_MG[chess.PAWN]
        else:
            victim_val = 0
            
        attacker = board.piece_at(move.from_square)
        attacker_val = Heuristics.MATERIAL_MG.get(attacker.piece_type, 0)
        
        # MVV (High) - LVA (Low)
        # e.g. PxQ = 900 - 100 + 10000 = 10800
        # QxP = 100 - 900 + 10000 = 9200
        return victim_val - attacker_val + 10000

    @staticmethod
    def quiescence_search(board, alpha, beta, color_sign):
        """
        Searches only captures to resolve tactical instability.
        color_sign: 1 for White, -1 for Black (Simulating Negamax perspective)
        Actually, let's stick to standard Negamax: 
        Maximize value for CURRENT player.
        """
        # Stand pat (Static Eval)
        # Score from perspective of side to move
        stand_pat = Heuristics.get_eval_score(board) * color_sign
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Generate captures
        moves = list(board.generate_legal_captures())
        # Sort MVV-LVA
        moves.sort(key=lambda m: Heuristics._mvv_lva_sort_key(board, m), reverse=True)
        
        for move in moves:
            board.push(move)
            score = -Heuristics.quiescence_search(board, -beta, -alpha, -color_sign)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha

    @staticmethod
    def negamax(board, depth, alpha, beta, color_sign):
        """
        Standard Alpha-Beta Search.
        returns (score, best_move)
        """
        if depth == 0:
            return Heuristics.quiescence_search(board, alpha, beta, color_sign), None

        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                return -20000, None # Mated (bad for us)
            return 0, None # Draw

        moves = list(board.legal_moves)
        # Basic sorting: Captures first
        moves.sort(key=lambda m: Heuristics._mvv_lva_sort_key(board, m), reverse=True)
        
        best_move = None
        best_score = -float('inf')
        
        for move in moves:
            board.push(move)
            score, _ = Heuristics.negamax(board, depth - 1, -beta, -alpha, -color_sign)
            score = -score
            board.pop()
            
            if score >= beta:
                return beta, move # Pruning
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
                
        return best_score, best_move

    @staticmethod
    def get_best_move_score(board, depth=1):
        """
        Returns normalized score [-1, 1] based on a search.
        Used by MCTS to 'verify' nodes.
        Warning: Depth > 1 might be slow in Python if called 1000s of times.
        Depth 1 + QSearch is very fast and effective.
        """
        color_sign = 1 if board.turn == chess.WHITE else -1
        # Run Negamax
        score, _ = Heuristics.negamax(board, depth, -20000, 20000, color_sign)
        
        # Output is from perspective of side to move.
        # But our Heuristics.evaluate returned White perspective usually?
        # mcts.py expects:
        #  If White Turn: Return +1 if White winning (Standard conventions vary).
        #  Let's look at mcts.py usage:
        #    heur_score = Heuristics.evaluate(board) 
        #    if turn == BLACK: heur_score = -heur_score (Converting to relative).
        # So Heuristics.evaluate was Absolute (White biased).
        
        # Negamax returns Relative Score (side to move).
        # So we just need to return it directly?
        # If board.turn == White, Score is White Adv.
        # If board.turn == Black, Score is Black Adv.
        
        # But MCTS `heur_score` logic does: `if black: -heur_score`.
        # It EXPECTS Absolute Score (from White perspective).
        # So if I return a Relative Score here, I need to convert it back to Absolute?
        # Or change MCTS.
        
        # Let's return Absolute (White Perspective) to be compatible with current MCTS.
        if board.turn == chess.BLACK:
            abs_score = -score
        else:
            abs_score = score
            
        # Normalize
        return max(-1.0, min(1.0, abs_score / 1000.0))
