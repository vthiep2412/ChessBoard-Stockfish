use chess::{Board, Color, Piece, Square, File, BitBoard};

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
        let index = (attack_units as usize).min(49);
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
pub fn evaluate_with_state(board: &Board, state: &EvalState, alpha: i32, beta: i32) -> i32 {
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

pub fn see(board: &Board, mv: chess::ChessMove) -> i32 {
    if !is_tactical(board, mv) {
        return 0;
    }

    // Handle Promotion Value in SEE
    if let Some(p) = mv.get_promotion() {
         let val = match p {
            Piece::Queen => 900,
            Piece::Rook => 500, 
            Piece::Bishop => 330,
            Piece::Knight => 320,
            _ => 0,
         };
         return val - 100; // Value minus Pawn
    }

    // Code Rabbit Fix: Handle missing victim properly
    let victim = match board.piece_on(mv.get_dest()) {
        Some(p) => p,
        None => {
            // Handle En Passant
            if board.en_passant() == Some(mv.get_dest()) && board.piece_on(mv.get_source()) == Some(Piece::Pawn) {
                Piece::Pawn
            } else {
                return 0; // Not a capture
            }
        }
    };
    let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);

    fn piece_val(p: Piece) -> i32 {
        match p {
            Piece::Pawn => 100, Piece::Knight => 320, Piece::Bishop => 330,
            Piece::Rook => 500, Piece::Queen => 900, Piece::King => 20000,
        }
    }

    piece_val(victim) - piece_val(attacker)
}
