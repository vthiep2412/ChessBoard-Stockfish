use chess::{Board, Color, Piece, Square, File, BitBoard};

// =============================================================================
// EVALUATION CONSTANTS (Stockfish 11 Style - Scaled to millipawns)
// =============================================================================

// Material values (mg, eg)
pub const PAWN_VAL: [i32; 2] = [128, 172];
pub const KNIGHT_VAL: [i32; 2] = [781, 854];
pub const BISHOP_VAL: [i32; 2] = [825, 915];
pub const ROOK_VAL: [i32; 2] = [1276, 1380];
pub const QUEEN_VAL: [i32; 2] = [2538, 2682];

// King Safety Constants
// Tuned values: Increased to punish open kings more severely (Range 0-25)
const ATTACK_WEIGHT: [i32; 5] = [0, 12, 12, 18, 30]; // Piece types: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
const SAFETY_TABLE_MAX: i32 = 900;
const SAFETY_TABLE: [i32; 100] = [
    0,   0,   0,   1,   3,   5,   7,   9,  12,  16,
   20,  24,  28,  33,  39,  45,  51,  57,  64,  72,
   80,  88,  96, 105, 115, 125, 135, 145, 156, 168,
  180, 192, 204, 217, 231, 245, 259, 273, 288, 304,
  320, 336, 352, 369, 387, 405, 423, 441, 460, 480,
  500, 520, 540, 561, 583, 605, 627, 649, 672, 696,
  720, 744, 768, 793, 819, 845, 871, 897, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX,
  SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX,
  SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX,
  SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX, SAFETY_TABLE_MAX,
];

// Pawn Structure
const PAWN_ISOLATED: i32 = -15;
const PAWN_PASSED: [i32; 8] = [0, 5, 10, 20, 40, 80, 150, 200]; // Bonus by rank
const LAZY_EVAL_MARGIN: i32 = 500;

// Mobility (Safe squares available)
const MOBILITY_BONUS: [i32; 5] = [0, 4, 3, 2, 1]; // Bonus per safe square for N, B, R, Q

// Incremental Update State
#[derive(Clone, Copy)]
pub struct EvalState {
    pub mg_material: [i32; 2], // [White, Black]
    pub eg_material: [i32; 2],
    pub pst_mg: [i32; 2],      // PST Scores [White, Black]
    pub pst_eg: [i32; 2],
    pub phase: i32,            // Game phase (24 = Start, 0 = End)
}

impl EvalState {
    pub fn new(board: &Board) -> Self {
        let mut s = Self {
            mg_material: [0, 0],
            eg_material: [0, 0],
            pst_mg: [0, 0],
            pst_eg: [0, 0],
            phase: 0,
        };
        
        // Initial phase calculation
        s.phase = game_phase(board);

        for sq in 0..64 {
            // Safety: sq is 0..64, so it is a valid square index.
            let square = unsafe { Square::new(sq as u8) };
            if let Some(piece) = board.piece_on(square) {
                let color = board.color_on(square).unwrap();
                s.add_material_only(color, piece, square);
            }
        }
        s
    }

    // Helper to add material + PST only (phase is pre-calc in new)
    fn add_material_only(&mut self, color: Color, piece: Piece, sq: Square) {
        let c_idx = color.to_index();
        let (mg, eg) = get_material(piece);
        let (pmg, peg) = crate::pst::get_pst(piece, sq, color);
        
        self.mg_material[c_idx] += mg;
        self.eg_material[c_idx] += eg;
        self.pst_mg[c_idx] += pmg;
        self.pst_eg[c_idx] += peg;
    }

    fn add_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let c_idx = color.to_index();
        let (mg, eg) = get_material(piece);
        let (pmg, peg) = crate::pst::get_pst(piece, sq, color);
        
        self.mg_material[c_idx] += mg;
        self.eg_material[c_idx] += eg;
        self.pst_mg[c_idx] += pmg;
        self.pst_eg[c_idx] += peg;
        self.phase = (self.phase + get_phase_value(piece)).min(24);
    }

    fn remove_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let c_idx = color.to_index();
        let (mg, eg) = get_material(piece);
        let (pmg, peg) = crate::pst::get_pst(piece, sq, color);
        
        self.mg_material[c_idx] -= mg;
        self.eg_material[c_idx] -= eg;
        self.pst_mg[c_idx] -= pmg;
        self.pst_eg[c_idx] -= peg;
        self.phase = (self.phase - get_phase_value(piece)).max(0);
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
                    let captured_sq_idx = if us == Color::White {
                        ep_sq.to_index() - 8
                    } else {
                        ep_sq.to_index() + 8
                    };

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
        // We must update the Rook's PST because it moves!
        // Check if the move is a castling move (King moved 2 squares)
        if piece == Piece::King && (source.to_index() as i8 - dest.to_index() as i8).abs() == 2 {
            // Determine castling type and rook squares
            let (rook_src, rook_dst) = if dest > source {
                 // Kingside (e.g. e1 -> g1, so rook h1 -> f1)
                 // e1=4, g1=6. h1=7, f1=5.
                 (unsafe { Square::new(source.to_index() as u8 + 3) }, unsafe { Square::new(source.to_index() as u8 + 1) })
            } else {
                 // Queenside (e.g. e1 -> c1, so rook a1 -> d1)
                 // e1=4, c1=2. a1=0, d1=3.
                 (unsafe { Square::new(source.to_index() as u8 - 4) }, unsafe { Square::new(source.to_index() as u8 - 1) })
            };
            
            // For Black (source=e8=60), same relative offsets apply?
            // Kingside: e8->g8 (60->62). Rook h8->f8 (63->61).
            // 60+3=63. 60+1=61. Correct.
            // Queenside: e8->c8 (60->58). Rook a8->d8 (56->59).
            // 60-4=56. 60-1=59. Correct.

            // Move the rook in EvalState
            self.remove_piece(us, Piece::Rook, rook_src);
            self.add_piece(us, Piece::Rook, rook_dst);
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

fn get_phase_value(piece: Piece) -> i32 {
    match piece {
        Piece::Knight | Piece::Bishop => 1,
        Piece::Rook => 2,
        Piece::Queen => 4,
        _ => 0,
    }
}

pub fn game_phase(board: &Board) -> i32 {
    let mut phase = 0;
    for sq in 0..64 {
         let square = unsafe { Square::new(sq as u8) };
         if let Some(piece) = board.piece_on(square) {
             phase += get_phase_value(piece);
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

fn rank_bb(rank: u8) -> BitBoard {
    BitBoard::new(0xFF << (rank * 8))
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

        // Proper passed pawn check
        if (passed_mask & enemy_pawns).0 == 0 {
            let r = if color == Color::White { rank.to_index() } else { 7 - rank.to_index() };
            // Base bonus
            let mut bonus = PAWN_PASSED[r];

            // Connected Bonus: Friendly pawns on adjacent files AND (Rank or Rank-1/Rank+1)
            let rank_idx = rank.to_index() as u8;

            // Calculate "rank behind" for support (White: r-1, Black: r+1)
            let behind_rank = if color == Color::White {
                rank_idx.checked_sub(1)
            } else {
                if rank_idx < 7 { Some(rank_idx + 1) } else { None }
            };

            let mut support_mask = rank_bb(rank_idx);
            if let Some(r) = behind_rank {
                support_mask |= rank_bb(r);
            }

            if (neighbors & pawns & support_mask).0 != 0 {
                bonus += bonus / 2; // 50% boost for connected passers
            }

            // Scale based on king proximity
            let king_sq = board.king_square(color);
            let enemy_king_sq = board.king_square(!color);

            let dist_us = distance(king_sq, sq);
            let dist_them = distance(enemy_king_sq, sq);

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
        let safe = moves & !*us; 
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

// FAST King Safety & Pawn Shield
fn eval_king_safety(board: &Board, color: Color) -> i32 {
    let us = board.color_combined(color);
    let them = board.color_combined(!color);
    let occ = *us | *them;
    let king_sq = board.king_square(color);
    let mut score = 0;

    // 1. Piece Attacks on King Zone
    let king_moves = chess::get_king_moves(king_sq);
    let zone = king_moves | BitBoard::from_square(king_sq);

    // Shift forward for "extended zone"
    let k_bb = BitBoard::from_square(king_sq);
    let f1 = if color == Color::White { k_bb.0 << 8 } else { k_bb.0 >> 8 };
    let f2 = if color == Color::White { k_bb.0 << 16 } else { k_bb.0 >> 16 };
    let full_zone = zone | BitBoard::new(f1) | BitBoard::new(f2);

    let mut attack_units = 0;
    let mut attackers_count = 0;

    for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let pieces = board.pieces(piece) & them;
        for sq in pieces {
            let attacks = attacks_by_piece(piece, sq, occ);
            if (attacks & full_zone).0 != 0 {
                attack_units += ATTACK_WEIGHT[piece.to_index()];
                attackers_count += 1;
            }
        }
    }

    // 2. Pawn Shield (Optimized Bitwise)
    // Shield squares: The 3 squares in front of the king
    let shield_mask = if color == Color::White {
        // Shift King left-up, up, right-up (needs overflow check or file mask)
        // Simplest: Just use generic shifts with file checks
        let k = king_sq.to_index();
        let mut mask = 0u64;
        if k < 56 { // Not on Rank 8 (impossible for valid king anyway)
            let up = k + 8;
            mask |= 1 << up; // Directly in front
            if king_sq.get_file().to_index() > 0 { mask |= 1 << (up - 1); } // Front-Left
            if king_sq.get_file().to_index() < 7 { mask |= 1 << (up + 1); } // Front-Right
        }
        BitBoard::new(mask)
    } else {
        // Black
        let k = king_sq.to_index();
        let mut mask = 0u64;
        if k > 7 { // Not on Rank 1
            let down = k - 8;
            mask |= 1 << down;
            if king_sq.get_file().to_index() > 0 { mask |= 1 << (down - 1); }
            if king_sq.get_file().to_index() < 7 { mask |= 1 << (down + 1); }
        }
        BitBoard::new(mask)
    };

    let our_pawns = board.pieces(Piece::Pawn) & us;
    let shield_count = (shield_mask & our_pawns).popcnt();

    // Penalty logic: 
    // Ideally 3 pawns. 
    // Missing 1: Small penalty
    // Missing 2: Bigger penalty
    // Missing 3: Huge penalty (Open King)
    let missing = 3 - shield_count.min(3);
    if missing > 0 {
        // Polynomial penalty: 0->0, 1->10, 2->30, 3->60
        score -= (missing * missing + missing) as i32 * 10; 
    }

    // Storm detection: Enemy pawns attacking shield?
    let _enemy_pawns = board.pieces(Piece::Pawn) & them;
    // Simple check: Enemy pawns on adjacent files and close rank?
    // TODO: Add storm logic later if needed (keeping it simple and fast for now)
    
    if attackers_count > 0 || attack_units > 0 {
        let index = (attack_units as usize).min(99);
        score -= SAFETY_TABLE[index];
    }

    score
}

fn eval_king_endgame(board: &Board, color: Color) -> i32 {
    let king_sq = board.king_square(color);
    let file = king_sq.get_file().to_index();
    let rank = king_sq.get_rank().to_index();

    let dist_file = (file as i32 - 3).abs().min((file as i32 - 4).abs());
    let dist_rank = (rank as i32 - 3).abs().min((rank as i32 - 4).abs());

    let dist = dist_file + dist_rank;
    (14 - dist) * 10
}

/// Main evaluation function
pub fn evaluate_with_state(board: &Board, state: &EvalState, _alpha: i32, _beta: i32) -> i32 {
    let us = board.side_to_move();
    let them = !us;

    // O(1) Phase from State
    let phase = state.phase;
    let mg_weight = phase.min(24);
    let eg_weight = 24 - mg_weight;
    
    // Material + PST
    let mut score_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()]
                     + state.pst_mg[us.to_index()] - state.pst_mg[them.to_index()];
    let mut score_eg = state.eg_material[us.to_index()] - state.eg_material[them.to_index()]
                     + state.pst_eg[us.to_index()] - state.pst_eg[them.to_index()];
    
    // Lazy Evaluation Removed for Accuracy
    // We trust our high NPS to handle full eval.
    // const LAZY_MARGIN: i32 = 150;
    // let stand_pat = (score_mg * mg_weight + score_eg * eg_weight) / 24;
    // if stand_pat - LAZY_MARGIN >= beta { return stand_pat; }
    // if stand_pat + LAZY_MARGIN <= alpha { return stand_pat; }

    // Full Evaluation
    // Pawn Structure
    let pawns_us = eval_pawns(board, us);
    let pawns_them = eval_pawns(board, them);
    score_mg += pawns_us - pawns_them;
    score_eg += (pawns_us - pawns_them) * 2;
    
    // Mobility
    let mob_us = eval_mobility(board, us);
    let mob_them = eval_mobility(board, them);
    score_mg += mob_us - mob_them;
    score_eg += mob_us - mob_them;

    // King Safety
    let safety_us = eval_king_safety(board, us);
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
    if board.side_to_move() == chess::Color::Black {
        score = -score;
    }
    score
}

pub fn evaluate_lazy(board: &Board, state: &EvalState, alpha: i32, beta: i32) -> i32 {
    let us = board.side_to_move();
    let them = !us;
    
    let score_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()]
                 + state.pst_mg[us.to_index()] - state.pst_mg[them.to_index()];
    let score_eg = state.eg_material[us.to_index()] - state.eg_material[them.to_index()]
                 + state.pst_eg[us.to_index()] - state.pst_eg[them.to_index()];

    // O(1) Phase from State
    let phase = state.phase;
    let mg_weight = phase.min(24);
    let eg_weight = 24 - mg_weight;

    let score = (score_mg * mg_weight + score_eg * eg_weight) / 24;

    // Relaxed margin to prevent pruning tactical sequences (sacrifices)
    let margin = LAZY_EVAL_MARGIN;
    if score < alpha - margin {
        return alpha;
    }
    if score > beta + margin {
        return beta;
    }

    score
}

// Helpers for Search
// Helper for Debugging
pub fn debug_eval(board: &Board) -> std::collections::HashMap<String, i32> {
    let mut map = std::collections::HashMap::new();
    let state = EvalState::new(board);
    let us = board.side_to_move();
    let them = !us;
    
    map.insert("phase".to_string(), state.phase);
    map.insert("mg_mat_us".to_string(), state.mg_material[us.to_index()]);
    map.insert("mg_mat_them".to_string(), state.mg_material[them.to_index()]);
    map.insert("eg_mat_us".to_string(), state.eg_material[us.to_index()]);
    map.insert("eg_mat_them".to_string(), state.eg_material[them.to_index()]);
    
    map.insert("pst_mg_us".to_string(), state.pst_mg[us.to_index()]);
    map.insert("pst_mg_them".to_string(), state.pst_mg[them.to_index()]);
    map.insert("pst_eg_us".to_string(), state.pst_eg[us.to_index()]);
    map.insert("pst_eg_them".to_string(), state.pst_eg[them.to_index()]);
    
    // Components
    let mat_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()];
    let pst_mg = state.pst_mg[us.to_index()] - state.pst_mg[them.to_index()];
    map.insert("diff_mat_mg".to_string(), mat_mg);
    map.insert("diff_pst_mg".to_string(), pst_mg);
    
    let score = evaluate_lazy(board, &state, -30000, 30000);
    map.insert("total_score".to_string(), score);
    
    map
}

pub fn is_capture(board: &Board, mv: chess::ChessMove) -> bool {
    board.piece_on(mv.get_dest()).is_some()
    || (board.en_passant() == Some(mv.get_dest()) && board.piece_on(mv.get_source()) == Some(chess::Piece::Pawn))
}

pub fn is_tactical(board: &Board, mv: chess::ChessMove) -> bool {
    is_capture(board, mv) || mv.get_promotion().is_some()
}

pub fn mvv_lva_score(board: &Board, mv: chess::ChessMove) -> i32 {
    // 1. Handle Promotions (Priority over most captures)
    if let Some(p) = mv.get_promotion() {
        return match p {
            Piece::Queen => 900,  // High value
            Piece::Rook => 500,
            Piece::Bishop => 330,
            Piece::Knight => 320,
            _ => 0,
        };
    }

    // 2. Handle Captures
    // Code Rabbit Fix: Don't mask missing victim with unwrap_or
    let victim = match board.piece_on(mv.get_dest()) {
        Some(p) => p,
        None => {
            // Handle En Passant (victim is pawn on adjacent file)
            if board.en_passant() == Some(mv.get_dest()) && board.piece_on(mv.get_source()) == Some(Piece::Pawn) {
                Piece::Pawn
            } else {
                // Not a capture
                return 0;
            }
        }
    };

    let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
    
    let v_val = match victim {
        Piece::Pawn => 1, Piece::Knight => 3, Piece::Bishop => 3,
        Piece::Rook => 5, Piece::Queen => 9, Piece::King => 100,
    };
    let a_val = match attacker {
        Piece::Pawn => 1, Piece::Knight => 3, Piece::Bishop => 3,
        Piece::Rook => 5, Piece::Queen => 9, Piece::King => 100,
    };
    
    // Score: Victim - Attacker (plus offset to keep positive)
    v_val * 10 - a_val + 100
}

/// Helper to get simple piece value (MG) for pruning
pub fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => PAWN_VAL[0],
        Piece::Knight => KNIGHT_VAL[0],
        Piece::Bishop => BISHOP_VAL[0],
        Piece::Rook => ROOK_VAL[0],
        Piece::Queen => QUEEN_VAL[0],
        Piece::King => 20000,
    }
}

// Helper to find the least valuable attacker for the given side
fn get_least_valuable_attacker(board: &Board, sq: Square, side: Color, occupied: BitBoard) -> Option<(Piece, Square)> {
    // 1. Pawns
    // Check pawns of 'side' that attack 'sq' AND are currently occupied (not captured)
    let pawns = board.pieces(Piece::Pawn) & board.color_combined(side) & occupied;
    if pawns.0 != 0 {
        // Attack vectors reversed:
        // If side is White, attackers are on squares that Black pawns at 'sq' would attack.
        let pawn_attacks = if side == Color::White {
             chess::get_pawn_attacks(sq, Color::Black, BitBoard::new(!0))
        } else {
             chess::get_pawn_attacks(sq, Color::White, BitBoard::new(!0))
        };

        let valid_pawns = pawn_attacks & pawns;
        if valid_pawns.0 != 0 {
             return Some((Piece::Pawn, valid_pawns.to_square())); // LSB
        }
    }

    // 2. Knights
    let knights = board.pieces(Piece::Knight) & board.color_combined(side) & occupied;
    if knights.0 != 0 {
        let attacks = chess::get_knight_moves(sq) & knights;
        if attacks.0 != 0 {
             return Some((Piece::Knight, attacks.to_square()));
        }
    }

    // 3. Bishops
    let bishops = board.pieces(Piece::Bishop) & board.color_combined(side) & occupied;
    if bishops.0 != 0 {
         let attacks = chess::get_bishop_moves(sq, occupied) & bishops;
         if attacks.0 != 0 { return Some((Piece::Bishop, attacks.to_square())); }
    }

    // 4. Rooks
    let rooks = board.pieces(Piece::Rook) & board.color_combined(side) & occupied;
    if rooks.0 != 0 {
         let attacks = chess::get_rook_moves(sq, occupied) & rooks;
         if attacks.0 != 0 { return Some((Piece::Rook, attacks.to_square())); }
    }

    // 5. Queens
    let queens = board.pieces(Piece::Queen) & board.color_combined(side) & occupied;
    if queens.0 != 0 {
         let attacks = (chess::get_bishop_moves(sq, occupied) | chess::get_rook_moves(sq, occupied)) & queens;
         if attacks.0 != 0 { return Some((Piece::Queen, attacks.to_square())); }
    }

    // 6. King
    // We explicitly EXCLUDE King captures in SEE to avoid illegal moves (moving into check).
    // SEE is an approximation, and King recaptures are rare in tactical sequences
    // where they don't immediately lose the game or aren't forced.
    // For correctness, we assume the King never recaptures in this static analysis.

    None
}

/// Static Exchange Evaluation (SEE)
/// Returns the approximate material gain of the move, handling all consequent captures.
pub fn see(board: &Board, mv: chess::ChessMove) -> i32 {
    if !is_tactical(board, mv) {
        return 0;
    }

    let mut scores = [0i32; 32];
    let mut d = 0;

    let from = mv.get_source();
    let to = mv.get_dest();
    let mut attacker_piece = board.piece_on(from).unwrap_or(Piece::Pawn);

    // Initial capture value
    let mut victim_val = if let Some(victim) = board.piece_on(to) {
        piece_value(victim)
    } else {
        // En Passant?
        if board.en_passant() == Some(to) && attacker_piece == Piece::Pawn {
             piece_value(Piece::Pawn)
        } else {
             0
        }
    };

    // Handle Promotion
    if let Some(promo) = mv.get_promotion() {
         let promo_val = piece_value(promo);
         let pawn_val = piece_value(Piece::Pawn);
         victim_val += promo_val - pawn_val;
         attacker_piece = promo;
    }

    scores[d] = victim_val;
    d += 1;

    let mut side = !board.side_to_move();
    let mut occupied = *board.combined();

    // Make the move on the bitboard (simulated)
    occupied ^= BitBoard::from_square(from);
    occupied |= BitBoard::from_square(to);

    // If EP, remove the captured pawn (on adjacent square)
    if board.en_passant() == Some(to) && attacker_piece == Piece::Pawn {
        // Safe EP capture square calculation
        let cap_rank = if board.side_to_move() == Color::White {
             // White moves up, captured pawn is behind 'to' (Rank 5 -> 4)
             chess::Rank::Fifth
        } else {
             // Black moves down, captured pawn is behind 'to' (Rank 4 -> 5)
             chess::Rank::Fourth
        };
        let cap_sq = Square::make_square(cap_rank, to.get_file());
        occupied ^= BitBoard::from_square(cap_sq);
    }

    loop {
        // Find LVA for 'side' attacking 'to'
        let lva = get_least_valuable_attacker(board, to, side, occupied);
        if let Some((p, sq)) = lva {
             scores[d] = piece_value(attacker_piece);

             // Handle Promotion in the chain
             // If the *next* attacker (p) is a pawn and moves to a promotion rank,
             // we update it to a Queen for subsequent calculations.
             let mut next_attacker = p;

             // Check if 'p' (the piece at 'sq') is a pawn that will promote upon capturing on 'to'.
             if p == Piece::Pawn {
                 let rank_of_capture = to.get_rank();
                 // If side is White, they are capturing on rank_of_capture.
                 // A White pawn promotes if it lands on Rank 8.
                 // A Black pawn promotes if it lands on Rank 1.

                 // 'side' here is the side that OWNS 'p' (the attacker we just found).
                 // In the loop, 'side' was flipped to !board.side_to_move() at start,
                 // then flipped again inside.
                 // Wait.
                 // In `see`:
                 // `let mut side = !board.side_to_move();` (The side being attacked? No.)
                 // `get_least_valuable_attacker(..., side, ...)`
                 // We want an attacker belonging to `side`.
                 // So `side` IS the side moving to capture.

                 if (side == Color::White && rank_of_capture == chess::Rank::Eighth) ||
                    (side == Color::Black && rank_of_capture == chess::Rank::First)
                 {
                     next_attacker = Piece::Queen;
                 }
             }

             // Prepare for next
             attacker_piece = next_attacker;
             side = !side;
             occupied ^= BitBoard::from_square(sq);
             d += 1;

             if d >= 31 { break; }
        } else {
             break;
        }
    }

    // Backpropagate Minimax
    // scores[0] is the gain of the FIRST capture (forced).
    // scores[1] is the value of the piece that made the first capture (risk for opponent).
    let mut score = 0;
    while d > 1 {
        d -= 1;
        score = (scores[d] - score).max(0);
    }

    scores[0] - score
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_see_pxq_protected() {
        // White Pawn on c4, Black Queen on d5. Black Pawn on c6 protecting Queen.
        // Added Kings at e1/e8 to make FEN valid.
        // FEN: 4k3/8/2p5/3q4/2P5/8/8/4K3 w - - 0 1
        // Move c4d5.
        // White takes Queen (900). Black takes Pawn (100). Net +800.
        // piece_value: Q=2538, P=128.
        // White takes Q (+2538). Black takes P (128). Net +2410.
        let board = Board::from_str("4k3/8/2p5/3q4/2P5/8/8/4K3 w - - 0 1").unwrap();
        let mv = chess::ChessMove::new(
            Square::make_square(chess::Rank::Fourth, chess::File::C),
            Square::make_square(chess::Rank::Fifth, chess::File::D),
            None
        );
        let score = see(&board, mv);
        println!("PxQ Protected SEE: {}", score);
        assert!(score > 2000);
    }

    #[test]
    fn test_see_qxp_protected() {
        // White Queen on d4. Black Pawn on d5. Black Pawn on c6 protecting.
        // Added Kings.
        // FEN: 4k3/8/2p5/3p4/3Q4/8/8/4K3 w - - 0 1
        // Move d4d5.
        // White takes P (128). Black takes Q (2538).
        // Net: 128 - 2538 = -2410.
        let board = Board::from_str("4k3/8/2p5/3p4/3Q4/8/8/4K3 w - - 0 1").unwrap();
        let mv = chess::ChessMove::new(
            Square::make_square(chess::Rank::Fourth, chess::File::D),
            Square::make_square(chess::Rank::Fifth, chess::File::D),
            None
        );
        let score = see(&board, mv);
        println!("QxP Protected SEE: {}", score);
        assert!(score < -2000);
    }

    #[test]
    fn test_see_battery_xray() {
        // White R on e1, Q on e2. Black P on e4. Protected by N on f6.
        // Added Kings at c1/c8.
        // FEN: 2k5/8/5n2/8/4p3/8/4Q3/2K1R3 w - - 0 1
        // Move Qxe4.
        // 1. QxP (+128).
        // 2. NxQ (+2538).
        // 3. RxN (+781).
        // Net: 128 - 2538 + 781 = -1629.
        let board = Board::from_str("2k5/8/5n2/8/4p3/8/4Q3/2K1R3 w - - 0 1").unwrap();
        let mv = chess::ChessMove::new(
            Square::make_square(chess::Rank::Second, chess::File::E),
            Square::make_square(chess::Rank::Fourth, chess::File::E),
            None
        );
        let score = see(&board, mv);
        println!("QxP (NxQ, RxN) SEE: {}", score);
        // Expect negative
        assert!(score < -1000);
    }

    #[test]
    fn test_see_king_recapture_omitted() {
        // White Queen captures Pawn at e4. Black King at e5 protects it.
        // FEN: 8/8/8/4k3/4p3/8/4Q3/4K3 w - - 0 1
        // Move Qxe4.
        // If King recaptures: 1. QxP (+128). 2. KxQ (+2538). Net -2410.
        // If King ignored: 1. QxP (+128). Stop. Net +128.
        // We expect +128 (positive) because King shouldn't recapture in SEE.
        let board = Board::from_str("8/8/8/4k3/4p3/8/4Q3/4K3 w - - 0 1").unwrap();
        let mv = chess::ChessMove::new(
            Square::make_square(chess::Rank::Second, chess::File::E),
            Square::make_square(chess::Rank::Fourth, chess::File::E),
            None
        );
        let score = see(&board, mv);
        println!("QxP (Protected by K) SEE: {}", score);
        assert!(score > 0);
    }

    #[test]
    fn test_see_promotion_in_chain() {
        // White Rook on a8. Black Pawn on a2. White King on h1. Black King on h8.
        // Black Pawn captures something on b1 and promotes to Queen.
        // Let's set up: White Knight on b1. Black Pawn on c2. White Rook on c1.
        // Move PxC (promotes). Recapture by Rook.
        // FEN: 7k/8/8/8/8/8/2p5/1NR4K b - - 0 1
        // Black moves c2b1Q.
        // 1. PxN (+781) + Promo (Queen-Pawn).
        // 2. RxQ (+2538).
        // Net: +781 + (2538 - 128) - 2538 = 781 - 128 = 653. (Approx)
        // Actually SEE calculates gain.
        // Initial move value: Captured Knight (781) + Promo (2410) = 3191.
        // Victim for next: Queen (2538).
        // Attacker: Rook.
        // Score: 3191 - 2538 = 653.

        // Let's test chain *after* initial.
        // White Pawn on a7, Black Rook on a8. White Rook on a1.
        // White moves a7a8Q.
        // 1. PxR (1276) + Promo (2410) = 3686.
        // Stop.

        // Need chain.
        // White Pawn on a7. Black R on a8. Black R on b8 (battery?). No.
        // White Pawn on a7. Black Rook on a8. White Rook on a1.
        // Move: a7a8Q (Capture R).
        // 1. PxR (1276) + Promo.
        // 2. ... wait, if White plays, it's fine.

        // Scenario: Black Knight on a8. White Pawn on a7. Black Rook on b8 defending a8.
        // White plays a7a8Q.
        // 1. PxN (781) + Promo (2410) = 3191.
        // 2. RxQ (2538).
        // Net: 3191 - 2538 = 653.
        // If we didn't update attacker to Queen, it would be PxN (781) + ...
        // 2. RxP (128).
        // Net: 3191 - 128 = 3063.
        // So correct logic (Queen recapture) gives LOWER score (653).
        let board = Board::from_str("nKr5/P7/8/8/8/8/8/4k3 w - - 0 1").unwrap();
        // The move we test is the FIRST move.
        // But SEE takes a move and evaluates its result.
        // The test scenario described "Black Pawn captures...".
        // Here we set up White to move.
        // We want to test a sequence deeper than 1.

        // Let's ensure the initial move is a capture/promo.
        // Board: White Pawn on a7. Black Rook on a8.
        // Attackers on a8: White Pawn(a7).
        // Attackers on a8 (after): Black Rook on b8? (Need to add one).

        // FEN: 1r5k/P7/8/8/8/8/8/4K3 w - - 0 1
        // White to move. PxR=Q.
        // 1. Gain: Rook (1276) + Promo (2410). = 3686.
        // 2. Black Recaptures with Rook (b8).
        //    Victim: Queen (2538).
        //    Attacker: Rook.
        //    Score: 3686 - 2538 = 1148.

        // If we ignored promo update, victim would be Pawn (128).
        // Score: 3686 - 128 = 3558.

        // So if correct, score is 1148. If broken, 3558.
        // 1148 < 2000. 3558 > 2000.

        // FEN: rr5k/P7/... (Black rooks on a8, b8)
        let board = Board::from_str("rr5k/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mv = chess::ChessMove::new(
            Square::make_square(chess::Rank::Seventh, chess::File::A),
            Square::make_square(chess::Rank::Eighth, chess::File::A),
            Some(Piece::Queen)
        );
        let score = see(&board, mv);
        println!("PxR=Q (RxQ) SEE: {}", score);
        assert!(score < 2000);
        assert!(score > 500);
    }
}
