use chess::{Board, Color, Piece, Square, Rank, File, BitBoard, BoardStatus};

// =============================================================================
// EVALUATION CONSTANTS (Stockfish 11 Style - Scaled to millipawns)
// =============================================================================

// Material values (mg, eg)
const PAWN_VAL: [i32; 2] = [128, 172];
const KNIGHT_VAL: [i32; 2] = [781, 854];
const BISHOP_VAL: [i32; 2] = [825, 915];
const ROOK_VAL: [i32; 2] = [1276, 1380];
const QUEEN_VAL: [i32; 2] = [2538, 2682];

// King Safety Constants
const ATTACK_WEIGHT: [i32; 5] = [0, 2, 2, 3, 5]; // Piece types: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
const SAFETY_TABLE: [i32; 100] = [
    0,  0,   1,   2,   3,   5,   7,   9,  12,  15,
   18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
   68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
  140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
  260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
  377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
  494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
  500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
  500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
  500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
];

// Pawn Structure
const PAWN_ISOLATED: i32 = -15;
const PAWN_PASSED: [i32; 8] = [0, 5, 10, 20, 35, 60, 100, 200]; // Bonus by rank

// Mobility (Safe squares available)
const MOBILITY_BONUS: [i32; 5] = [0, 4, 3, 2, 1]; // Bonus per safe square for N, B, R, Q

// Incremental Update State
#[derive(Clone, Copy)]
pub struct EvalState {
    pub mg_material: [i32; 2], // [White, Black]
    pub eg_material: [i32; 2],
}

impl EvalState {
    pub fn new(board: &Board) -> Self {
        let mut s = Self {
            mg_material: [0, 0],
            eg_material: [0, 0],
        };
        
        for sq in 0..64 {
            // Safety: sq is 0..64, so it is a valid square index.
            let square = unsafe { Square::new(sq as u8) };
            if let Some(piece) = board.piece_on(square) {
                let color = board.color_on(square).unwrap();
                s.add_piece(color, piece, square);
            }
        }
        s
    }

    fn add_piece(&mut self, color: Color, piece: Piece, _sq: Square) {
        let c_idx = color.to_index();
        let (mg, eg) = get_material(piece);
        self.mg_material[c_idx] += mg;
        self.eg_material[c_idx] += eg;
    }

    fn remove_piece(&mut self, color: Color, piece: Piece, _sq: Square) {
        let c_idx = color.to_index();
        let (mg, eg) = get_material(piece);
        self.mg_material[c_idx] -= mg;
        self.eg_material[c_idx] -= eg;
    }
    
    /// Incremental update of evaluation state
    /// `board` must be the state BEFORE the move was applied
    pub fn apply_move(&mut self, board: &Board, mv: chess::ChessMove) {
        let us = board.side_to_move();
        let source = mv.get_source();
        let dest = mv.get_dest();

        // We assume the move is legal and the piece exists
        let piece = board.piece_on(source).unwrap();

        // 1. Remove moving piece from source
        self.remove_piece(us, piece, source);

        // 2. Handle Capture (Regular)
        if let Some(captured) = board.piece_on(dest) {
            self.remove_piece(!us, captured, dest);
        }

        // 3. Handle En Passant Capture
        if piece == Piece::Pawn {
            if let Some(ep_sq) = board.en_passant() {
                if dest == ep_sq {
                    // Captured pawn is implied
                    // The pawn is NOT on the EP square, but one rank "behind" it (relative to the capture).
                    // If we are White, we captured on Rank 6, so pawn was on Rank 5. (Index - 8)
                    // If we are Black, we captured on Rank 3, so pawn was on Rank 4. (Index + 8)

                    let captured_sq_idx = if us == Color::White {
                        ep_sq.to_index() - 8
                    } else {
                        ep_sq.to_index() + 8
                    };

                    // Safety: ep_sq is valid, and EP rules guarantee the pawn is there.
                    let captured_sq = unsafe { Square::new(captured_sq_idx as u8) };
                    self.remove_piece(!us, Piece::Pawn, captured_sq);
                }
            }
        }

        // 4. Place piece at dest (Handle Promotion)
        if let Some(promo) = mv.get_promotion() {
            self.add_piece(us, promo, dest);
        } else {
            self.add_piece(us, piece, dest);
        }

        // 5. Handle Castling (Rook move)
        if piece == Piece::King {
             let diff = (source.get_file().to_index() as i8 - dest.get_file().to_index() as i8).abs();
             if diff > 1 {
                 // Castling!
                 // We don't need exact squares for material update, but we need to know we moved a rook.
                 // Actually, we just move a rook from corner to next to king.
                 // Material doesn't change for the rook!
                 // Since we don't track PSQT in EvalState yet, we do NOTHING for the rook here.
                 // (Material score is invariant under movement).
             }
        }
    }
}

fn get_material(piece: Piece) -> (i32, i32) {
    match piece {
        Piece::Pawn => (PAWN_VAL[0], PAWN_VAL[1]),
        Piece::Knight => (KNIGHT_VAL[0], KNIGHT_VAL[1]),
        Piece::Bishop => (BISHOP_VAL[0], BISHOP_VAL[1]),
        Piece::Rook => (ROOK_VAL[0], ROOK_VAL[1]),
        Piece::Queen => (QUEEN_VAL[0], QUEEN_VAL[1]),
        Piece::King => (0, 0),
    }
}

pub fn game_phase(board: &Board) -> i32 {
    let mut phase = 0;
    for sq in 0..64 {
         // Safety: sq is 0..64, so it is a valid square index.
         let square = unsafe { Square::new(sq as u8) };
         if let Some(piece) = board.piece_on(square) {
             phase += match piece {
                 Piece::Knight | Piece::Bishop => 1,
                 Piece::Rook => 2,
                 Piece::Queen => 4,
                 _ => 0,
             };
         }
    }
    phase // 0 (Endgame) to 24 (Startpos)
}

// =============================================================================
// BITBOARD HELPERS
// =============================================================================

fn file_bb(sq: Square) -> BitBoard {
    let file = sq.get_file().to_index();
    let mut bb = 0x0101010101010101u64;
    bb <<= file;
    BitBoard::new(bb)
}

// Squares in front of the pawn (for passed pawn check)
fn front_span(color: Color, sq: Square) -> BitBoard {
    let bb = file_bb(sq);
    let rank = sq.get_rank().to_index();

    // Guard against edge shifts to avoid overflow
    if color == Color::White && rank == 7 {
        return BitBoard::new(0);
    }
    if color == Color::Black && rank == 0 {
        return BitBoard::new(0);
    }

    let mask = if color == Color::White {
        !((1u64 << (8 * (rank + 1))) - 1)
    } else {
        (1u64 << (8 * rank)) - 1
    };
    BitBoard::new(bb.0 & mask)
}

// Attacks by piece type (simplified lookup or generation)
fn attacks_by_piece(piece: Piece, sq: Square, occ: BitBoard) -> BitBoard {
    match piece {
        Piece::Knight => chess::get_knight_moves(sq),
        Piece::Bishop => chess::get_bishop_moves(sq, occ),
        Piece::Rook => chess::get_rook_moves(sq, occ),
        Piece::Queen => chess::get_bishop_moves(sq, occ) | chess::get_rook_moves(sq, occ),
        Piece::King => chess::get_king_moves(sq),
        _ => BitBoard::new(0),
    }
}

// =============================================================================
// EVALUATION TERMS
// =============================================================================

fn eval_pawns(board: &Board, color: Color) -> i32 {
    let us = board.color_combined(color);
    let them = board.color_combined(!color);
    let pawns = board.pieces(Piece::Pawn) & us;
    let enemy_pawns = board.pieces(Piece::Pawn) & them;
    let mut score = 0;

    for sq in pawns {
        let file = sq.get_file();
        let rank = sq.get_rank();

        // Isolated
        let mut neighbors = BitBoard::new(0);
        let _rank_idx = rank.to_index();
        let file_idx = file.to_index();

        // Safe square construction using make_square (Rank, File)
        if file_idx > 0 {
            let file_left = File::from_index(file_idx - 1);
            let s = Square::make_square(rank, file_left);
            neighbors |= file_bb(s);
        }
        if file_idx < 7 {
            let file_right = File::from_index(file_idx + 1);
            let s = Square::make_square(rank, file_right);
            neighbors |= file_bb(s);
        }

        if (neighbors & pawns).0 == 0 {
            score += PAWN_ISOLATED;
        }

        // Passed
        let front = front_span(color, sq);

        let mut passed_mask = front;
        if file_idx > 0 {
            let file_left = File::from_index(file_idx - 1);
            let s = Square::make_square(rank, file_left);
            passed_mask |= front_span(color, s);
        }
        if file_idx < 7 {
            let file_right = File::from_index(file_idx + 1);
            let s = Square::make_square(rank, file_right);
            passed_mask |= front_span(color, s);
        }

        // Proper passed pawn check: No enemy pawns in front or on adjacent files in front
        if (passed_mask & enemy_pawns).0 == 0 {
            let r = if color == Color::White { rank.to_index() } else { 7 - rank.to_index() };

            // Base bonus
            let mut bonus = PAWN_PASSED[r];

            // Scale based on king proximity (endgame logic)
            // If friendly king is close, good. If enemy king is far, good.
            let king_sq = board.king_square(color);
            let enemy_king_sq = board.king_square(!color);

            let dist_us = distance(king_sq, sq);
            let dist_them = distance(enemy_king_sq, sq);

            // Simple heuristic: 5 points per square advantage
            bonus += (dist_them as i32 - dist_us as i32) * 5;

            score += bonus;
        }
    }
    score
}

fn distance(sq1: Square, sq2: Square) -> u8 {
    let file_diff = (sq1.get_file().to_index() as i8 - sq2.get_file().to_index() as i8).abs();
    let rank_diff = (sq1.get_rank().to_index() as i8 - sq2.get_rank().to_index() as i8).abs();
    file_diff.max(rank_diff) as u8
}

fn eval_mobility(board: &Board, color: Color) -> i32 {
    let us = board.color_combined(color);
    let them = board.color_combined(!color);
    let occ = *us | *them;

    let mut score = 0;

    // Knights
    for sq in board.pieces(Piece::Knight) & us {
        let moves = chess::get_knight_moves(sq);
        let safe = moves & !*us; // Legal moves (capture or quiet)
        score += safe.popcnt() as i32 * MOBILITY_BONUS[1];
    }

    // Bishops
    for sq in board.pieces(Piece::Bishop) & us {
        let moves = chess::get_bishop_moves(sq, occ);
        let safe = moves & !*us;
        score += safe.popcnt() as i32 * MOBILITY_BONUS[2];
    }

    // Rooks
    for sq in board.pieces(Piece::Rook) & us {
        let moves = chess::get_rook_moves(sq, occ);
        let safe = moves & !*us;
        score += safe.popcnt() as i32 * MOBILITY_BONUS[3];
    }

    // Queens
    for sq in board.pieces(Piece::Queen) & us {
        let moves = chess::get_bishop_moves(sq, occ) | chess::get_rook_moves(sq, occ);
        let safe = moves & !*us;
        score += safe.popcnt() as i32 * MOBILITY_BONUS[4];
    }

    score
}

fn eval_pawn_shield(board: &Board, color: Color, king_sq: Square) -> i32 {
    let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
    let rank = king_sq.get_rank();
    let file_idx = king_sq.get_file().to_index();

    let shield_rank = if color == Color::White {
        if rank.to_index() >= 7 { return 0; }
        Rank::from_index(rank.to_index() + 1)
    } else {
        if rank.to_index() <= 0 { return 0; }
        Rank::from_index(rank.to_index() - 1)
    };

    let mut penalty = 0;

    // Check files: file-1, file, file+1
    for f in (file_idx.saturating_sub(1))..=(file_idx.saturating_add(1)).min(7) {
        let file = File::from_index(f);
        let shield_sq = Square::make_square(shield_rank, file);

        // Check for pawn on shield square
        // Or one rank further (push)
        // Simple logic: Check immediate shield square and one forward
        let sq1 = shield_sq;
        let sq2_rank = if color == Color::White {
             if shield_rank.to_index() < 7 { Some(Rank::from_index(shield_rank.to_index() + 1)) } else { None }
        } else {
             if shield_rank.to_index() > 0 { Some(Rank::from_index(shield_rank.to_index() - 1)) } else { None }
        };

        let mut found = false;
        if (BitBoard::from_square(sq1) & pawns).0 != 0 {
            found = true;
        } else if let Some(r2) = sq2_rank {
            let sq2 = Square::make_square(r2, file);
            if (BitBoard::from_square(sq2) & pawns).0 != 0 {
                found = true;
            }
        }

        if !found {
            // Missing shield pawn
            penalty += 20;

            // Check for open file (no pawns of either color)
            let all_pawns = board.pieces(Piece::Pawn);
            // Re-use file_bb logic locally for file 'f'
            let mut bb_val = 0x0101010101010101u64;
            bb_val <<= f;
            let file_mask = BitBoard::new(bb_val);

            if (file_mask & all_pawns).0 == 0 {
                penalty += 30; // Fully open file near king
            }
        }
    }

    -penalty
}

fn eval_king_safety(board: &Board, color: Color) -> i32 {
    let us = board.color_combined(color);
    let them = board.color_combined(!color);
    let occ = *us | *them;
    let king_sq = board.king_square(color);
    let mut score = 0;

    // King Zone: Squares around king + 2 squares in front
    let king_moves = chess::get_king_moves(king_sq);
    let mut zone = king_moves | BitBoard::from_square(king_sq);

    // Add squares in front (e.g. if White King at G1, add G2, H2, F2, maybe G3)
    // Simplified: Use the king ring plus forward push

    // Safety: Manual shifting, need to be careful with overflow/underflow logic (though wrapping handles it for bits)
    // Using BitBoard specific shifts is safer if available, but chess crate uses u64
    let k_bb = BitBoard::from_square(king_sq);
    // Shift forward 1 and 2 ranks
    let f1 = if color == Color::White { k_bb.0 << 8 } else { k_bb.0 >> 8 };
    let f2 = if color == Color::White { k_bb.0 << 16 } else { k_bb.0 >> 16 };

    // Combine into a danger zone (roughly) around the king and in front
    zone = zone | BitBoard::new(f1) | BitBoard::new(f2);

    let mut attack_units = 0;
    let mut attackers_count = 0;

    // Check enemy pieces attacking the zone
    for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let pieces = board.pieces(piece) & them;
        for sq in pieces {
            let attacks = attacks_by_piece(piece, sq, occ);

            // Attack on Zone
            if (attacks & zone).0 != 0 {
                attack_units += ATTACK_WEIGHT[piece.to_index()];
                attackers_count += 1;

                // Bonus weight for "Safe Checks" could go here
            }
        }
    }

    // Pawn Storm / Open File penalty
    // If king is on a file with no friendly pawns in front, it's open/semi-open
    let file = king_sq.get_file();
    let file_mask = file_bb(king_sq);
    let friendly_pawns = board.pieces(Piece::Pawn) & us;

    if (friendly_pawns & file_mask).0 == 0 {
        attack_units += 5; // Open file penalty
    }

    if attackers_count > 0 || attack_units > 0 {
        // Use lookup table
        let index = (attack_units as usize * attackers_count).min(99);
        score -= SAFETY_TABLE[index]; // Penalty for us (being attacked)
    }

    score
}

// Encourages king to centralize in endgame
fn eval_king_endgame(board: &Board, color: Color) -> i32 {
    let king_sq = board.king_square(color);
    let file = king_sq.get_file().to_index();
    let rank = king_sq.get_rank().to_index();

    // Distance from center (3.5, 3.5)
    // Dist from (3,3), (3,4), (4,3), (4,4)
    // Center squares are E4(4,3), D4(3,3), E5(4,4), D5(3,4)
    // Simply: max(|file-3.5|, |rank-3.5|) minimized
    // In integer math: sum of distance to center ranks/files

    let dist_file = (file as i32 - 3).abs().min((file as i32 - 4).abs());
    let dist_rank = (rank as i32 - 3).abs().min((rank as i32 - 4).abs());

    let dist = dist_file + dist_rank;

    // Bonus for being close to center: (MaxDist - Dist) * Weight
    (14 - dist) * 10
}

/// Main evaluation function
pub fn evaluate_with_state(board: &Board, state: &EvalState, alpha: i32, beta: i32) -> i32 {
    let us = board.side_to_move();
    let them = !us;

    let phase = game_phase(board);
    let mg_weight = phase.min(24);
    let eg_weight = 24 - mg_weight;
    
    let mut score_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()];
    let mut score_eg = state.eg_material[us.to_index()] - state.eg_material[them.to_index()];
    
    // Lazy Evaluation (Fast Path)
    // If material score is significantly outside alpha/beta, return early
    const LAZY_MARGIN: i32 = 150;
    let stand_pat = (score_mg * mg_weight + score_eg * eg_weight) / 24;

    if stand_pat - LAZY_MARGIN >= beta {
        return stand_pat;
    }
    if stand_pat + LAZY_MARGIN <= alpha {
        return stand_pat;
    }

    // Full Evaluation (Pawns, Mobility, Safety)
    
    // Pawn Structure
    let pawns_us = eval_pawns(board, us);
    let pawns_them = eval_pawns(board, them);
    score_mg += pawns_us - pawns_them;
    score_eg += (pawns_us - pawns_them) * 2; // Pawns more important in endgame?
    
    // Mobility
    let mob_us = eval_mobility(board, us);
    let mob_them = eval_mobility(board, them);
    score_mg += mob_us - mob_them;
    score_eg += mob_us - mob_them;

    // King Safety (Only MG usually)
    let safety_us = eval_king_safety(board, us); // Negative value (penalty)
    let safety_them = eval_king_safety(board, them);
    score_mg += safety_us - safety_them;

    // Endgame King Centralization
    let king_us = eval_king_endgame(board, us);
    let king_them = eval_king_endgame(board, them);
    score_eg += king_us - king_them;
    
    // Interpolate
    let score = (score_mg * mg_weight + score_eg * eg_weight) / 24;

    // Tempo bonus
    let tempo = 20;
    
    score + tempo
}

pub fn evaluate(board: &Board) -> i32 {
    let state = EvalState::new(board);
    let mut score = evaluate_with_state(board, &state, -30000, 30000);
    // Public API expects White-relative score, but evaluate_with_state returns STM-relative
    if board.side_to_move() == chess::Color::Black {
        score = -score;
    }
    score
}

pub fn evaluate_lazy(board: &Board, alpha: i32, beta: i32) -> i32 {
    let state = EvalState::new(board);
    let us = board.side_to_move();
    let them = !us;
    // Material only for lazy
    let score_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()];
    let score_eg = state.eg_material[us.to_index()] - state.eg_material[them.to_index()];

    let phase = game_phase(board);
    let mg_weight = phase.min(24);
    let eg_weight = 24 - mg_weight;

    let score = (score_mg * mg_weight + score_eg * eg_weight) / 24;

    // Margin-based early exit (fail-soft)
    // If the material score is way off, return bound
    let margin = 200;
    if score < alpha - margin {
        return alpha;
    }
    if score > beta + margin {
        return beta;
    }

    score
}

// Helpers for Search
pub fn is_capture(board: &Board, mv: chess::ChessMove) -> bool {
    // Check standard capture, promotion, or En Passant
    // Note: We treat promotions as captures for search pruning purposes (high value events)
    board.piece_on(mv.get_dest()).is_some()
    || mv.get_promotion().is_some()
    || (board.en_passant() == Some(mv.get_dest()) && board.piece_on(mv.get_source()) == Some(chess::Piece::Pawn))
}

pub fn mvv_lva_score(board: &Board, mv: chess::ChessMove) -> i32 {
    let victim = board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn); // En passant is pawn
    let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
    
    let v_val = match victim {
        Piece::Pawn => 1, Piece::Knight => 3, Piece::Bishop => 3,
        Piece::Rook => 5, Piece::Queen => 9, Piece::King => 100,
    };
    let a_val = match attacker {
        Piece::Pawn => 1, Piece::Knight => 3, Piece::Bishop => 3,
        Piece::Rook => 5, Piece::Queen => 9, Piece::King => 100,
    };
    
    v_val * 10 - a_val
}

// Static Exchange Evaluation (SEE) - LVA Approximation
pub fn see(board: &Board, mv: chess::ChessMove) -> i32 {
    if !is_capture(board, mv) {
        return 0;
    }

    let victim = board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
    let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);

    fn piece_val(p: Piece) -> i32 {
        match p {
            Piece::Pawn => 100, Piece::Knight => 320, Piece::Bishop => 330,
            Piece::Rook => 500, Piece::Queen => 900, Piece::King => 20000,
        }
    }

    // Simple LVA: Victim - Attacker (Optimistic)
    piece_val(victim) - piece_val(attacker)
}
