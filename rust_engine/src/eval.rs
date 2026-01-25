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
const PAWN_BACKWARD: i32 = -20;
const PAWN_DOUBLED: i32 = -10;
const PAWN_CONNECTED: i32 = 10;
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
    
    pub fn apply_move(&mut self, board: &Board, _mv: chess::ChessMove) {
        // Optimization: For now, just re-scan. It's safe.
        *self = Self::new(board);
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
        let rank_idx = rank.to_index();
        let file_idx = file.to_index();
        if file_idx > 0 {
            let s = unsafe { Square::new((rank_idx * 8 + (file_idx - 1)) as u8) };
            neighbors |= file_bb(s);
        }
        if file_idx < 7 {
            let s = unsafe { Square::new((rank_idx * 8 + (file_idx + 1)) as u8) };
            neighbors |= file_bb(s);
        }

        if (neighbors & pawns).0 == 0 {
            score += PAWN_ISOLATED;
        }

        // Backward (simplified)
        // No friendly pawns behind or on same rank on adjacent files
        // This is complex, skipping for now to focus on Passed

        // Passed
        // No enemy pawns in front on same file or adjacent files
        let front = front_span(color, sq);
        let mut span = front;
        // Expand span to adjacent files
        // Shift left/right
        // BitBoard doesn't impl simple shift, use raw u64
        let bb_span = front.0;
        let adj_mask = 0xFEFEFEFEFEFEFEFEu64; // Not A file
        let adj_mask_h = 0x7F7F7F7F7F7F7F7Fu64; // Not H file
        let left_span = (bb_span & adj_mask) << 1; // Wait, shifting logic depends on endian/mapping?
        // chess crate: 0=A1, 7=H1.
        // Left (A->B) is +1? No, A1=0, B1=1.
        // So file A is 0.
        // To check adjacent files, we mask and shift.
        // We can just use file_bb logic.

        let mut passed_mask = front;
        if file_idx > 0 {
            let s = unsafe { Square::new((rank_idx * 8 + (file_idx - 1)) as u8) };
            passed_mask |= front_span(color, s);
        }
        if file_idx < 7 {
            let s = unsafe { Square::new((rank_idx * 8 + (file_idx + 1)) as u8) };
            passed_mask |= front_span(color, s);
        }

        if (passed_mask & enemy_pawns).0 == 0 {
            // Passed!
            let r = if color == Color::White { rank.to_index() } else { 7 - rank.to_index() };
            score += PAWN_PASSED[r];
        }
    }
    score
}

fn eval_mobility(board: &Board, color: Color) -> i32 {
    let us = board.color_combined(color);
    let them = board.color_combined(!color);
    let occ = *us | *them;

    // Enemy pawn attacks (squares to avoid)
    // We need pawn attacks for !color
    // chess crate doesn't expose `get_pawn_attacks` easily for bitboards?
    // We can iterate enemy pawns.
    let enemy_pawns = board.pieces(Piece::Pawn) & them;
    let mut danger_zone = BitBoard::new(0);
    // Rough approximation: just dont count squares attacked by pawns
    // Skipping precise pawn attacks for speed in this step

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

fn eval_king_safety(board: &Board, color: Color) -> i32 {
    let us = board.color_combined(color);
    let them = board.color_combined(!color);
    let occ = *us | *them;
    let king_sq = board.king_square(color);

    // King Zone: Squares around king + maybe front
    let king_moves = chess::get_king_moves(king_sq);
    let mut zone = king_moves | BitBoard::from_square(king_sq);

    // Add squares in front (Rank+1/Rank-1)
    // Simplified: just surrounding 8 squares

    let mut attack_units = 0;
    let mut attackers_count = 0;

    // Check enemy pieces attacking the zone
    // Iterate enemy pieces (except pawns, usually handled separately)
    for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let pieces = board.pieces(piece) & them;
        for sq in pieces {
            let attacks = attacks_by_piece(piece, sq, occ);
            if (attacks & zone).0 != 0 {
                attack_units += ATTACK_WEIGHT[piece.to_index()];
                attackers_count += 1;
            }
        }
    }

    if attackers_count > 1 {
        // Use lookup table
        let index = (attack_units as usize * attackers_count).min(99);
        -SAFETY_TABLE[index] // Penalty for us (being attacked)
    } else {
        0
    }
}

/// Main evaluation function
pub fn evaluate_with_state(board: &Board, state: &EvalState, _alpha: i32, _beta: i32) -> i32 {
    let us = board.side_to_move();
    let them = !us;

    let phase = game_phase(board);
    let mg_weight = phase.min(24);
    let eg_weight = 24 - mg_weight;
    
    let mut score_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()];
    let mut score_eg = state.eg_material[us.to_index()] - state.eg_material[them.to_index()];
    
    // Evaluation Logic
    // Positive = Good for Side To Move (Relative)
    // But `evaluate` usually returns absolute score from White perspective or relative?
    // Rust chess convention: usually relative. But my `negamax` assumes relative.
    // However, material array is [White, Black].
    // So score_mg is (Us - Them). This is relative.
    
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
    score_mg += safety_us - safety_them; // My safety is bad, enemy safety is bad (so -(-x) = +x good for me)
    
    // Interpolate
    let score = (score_mg * mg_weight + score_eg * eg_weight) / 24;

    // Tempo bonus
    let tempo = 20;
    
    score + tempo
}

pub fn evaluate(board: &Board) -> i32 {
    let state = EvalState::new(board);
    evaluate_with_state(board, &state, -30000, 30000)
}

pub fn evaluate_lazy(board: &Board, _alpha: i32, _beta: i32) -> i32 {
    let state = EvalState::new(board);
    let us = board.side_to_move();
    let them = !us;
    // Material only for lazy
    let score_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()];
    let score_eg = state.eg_material[us.to_index()] - state.eg_material[them.to_index()];

    let phase = game_phase(board);
    let mg_weight = phase.min(24);
    let eg_weight = 24 - mg_weight;

    (score_mg * mg_weight + score_eg * eg_weight) / 24
}

// Helpers for Search
pub fn is_capture(board: &Board, mv: chess::ChessMove) -> bool {
    board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() // Treat promo as "interesting"
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

// Static Exchange Evaluation (SEE)
pub fn see(board: &Board, mv: chess::ChessMove) -> i32 {
    // Placeholder for SEE (complex)
    // If capture, return value. If quiet, return 0.
    if is_capture(board, mv) {
        let victim = board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
        let v_val = match victim {
            Piece::Pawn => 100, Piece::Knight => 320, Piece::Bishop => 330,
            Piece::Rook => 500, Piece::Queen => 900, Piece::King => 20000,
        };
        v_val // Very dumb SEE
    } else {
        0
    }
}
