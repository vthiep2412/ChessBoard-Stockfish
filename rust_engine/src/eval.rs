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

// PSQTs (Piece-Square Tables) - Simplified PeSTO tables for now, could be Tuned
// [rank][file] from White's perspective
// ... (Omitted full tables for brevity, using simple tables or keeping existing ones if valid)
// We will reuse the existing PSQTs from previous implementation or define new ones if needed.
// For "Stockfish 11 Style", we want detailed term-based evaluation.

// King Safety Constants
const ATTACK_WEIGHT: [i32; 5] = [0, 2, 2, 3, 5]; // Knight, Bishop, Rook, Queen, King? No, piece types.
// Piece types: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
// Weighted count of attackers
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
// [piece_type][safe_squares]
// Simplified: linear bonus
const MOBILITY_BONUS: [i32; 5] = [0, 4, 3, 2, 1]; // Bonus per safe square for N, B, R, Q

// Incremental Update State
#[derive(Clone, Copy)]
pub struct EvalState {
    pub mg_material: [i32; 2], // [White, Black]
    pub eg_material: [i32; 2],
    pub mg_psqt: [i32; 2],
    pub eg_psqt: [i32; 2],
}

impl EvalState {
    pub fn new(board: &Board) -> Self {
        // Full recalculation
        let mut s = Self {
            mg_material: [0, 0],
            eg_material: [0, 0],
            mg_psqt: [0, 0],
            eg_psqt: [0, 0],
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

    fn add_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let c_idx = color.to_index();
        let p_idx = piece.to_index();
        
        // Material
        let (mg, eg) = get_material(piece);
        self.mg_material[c_idx] += mg;
        self.eg_material[c_idx] += eg;
        
        // PSQT (Placeholder)
        // self.mg_psqt[c_idx] += PSQT_MG[p_idx][sq_idx];
        // self.eg_psqt[c_idx] += PSQT_EG[p_idx][sq_idx];
    }
    
    fn remove_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let c_idx = color.to_index();
        // let p_idx = piece.to_index();
        
        let (mg, eg) = get_material(piece);
        self.mg_material[c_idx] -= mg;
        self.eg_material[c_idx] -= eg;
    }
    
    pub fn apply_move(&mut self, board: &Board, mv: chess::ChessMove) {
        // This is tricky because 'board' is already updated in search?
        // No, typically we call this BEFORE making the move, or we need the OLD board?
        // Or we just recalculate for now to be safe (slow but correct).
        // Incremental update requires knowing what was captured.
        // For simplicity in this "Architecture Overhaul", let's use Lazy Recalculation
        // or just accept full recalc cost until we optimize eval.

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
    // Count non-pawn material
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

// Bitboard helpers
fn front_span(color: Color, sq: Square) -> BitBoard {
    // Squares in front of the pawn
    // Implementation depends on BitBoard ops
    // Placeholder
    BitBoard::new(0)
}

fn file_bb(sq: Square) -> BitBoard {
    // BitBoard of the file
    // Placeholder
    BitBoard::new(0)
}

/// Main evaluation function
pub fn evaluate_with_state(board: &Board, state: &EvalState, alpha: i32, beta: i32) -> i32 {
    evaluate(board) // Fallback to full eval for now
}

pub fn evaluate(board: &Board) -> i32 {
    let state = EvalState::new(board);
    let us = board.side_to_move();
    let them = !us;
    
    let phase = game_phase(board);
    let mg_weight = phase.min(24);
    let eg_weight = 24 - mg_weight;
    
    let mut score_mg = state.mg_material[us.to_index()] - state.mg_material[them.to_index()];
    let mut score_eg = state.eg_material[us.to_index()] - state.eg_material[them.to_index()];
    
    // ==========================================
    // PAWN STRUCTURE
    // ==========================================
    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    
    // Evaluate White Pawns
    let mut wp_score = 0;
    // Iterate pawns (placeholder logic)
    // for sq in white_pawns { ... }
    
    // ==========================================
    // KING SAFETY
    // ==========================================
    // Only calculate if King is exposed and opponent has pieces
    let mut safety_score = 0;
    // ...
    
    // Interpolate
    let score = (score_mg * mg_weight + score_eg * eg_weight) / 24;
    
    // Tempo bonus
    let tempo = 20;
    
    score + tempo
}

pub fn evaluate_lazy(board: &Board, alpha: i32, beta: i32) -> i32 {
    evaluate(board)
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
