//! PeSTO Piece-Square Tables and Evaluation
//! Ported from the Python implementation

use chess::{Board, Color, Piece, Square, ALL_SQUARES};

// Material values (middlegame)
const PAWN_MG: i32 = 82;
const KNIGHT_MG: i32 = 337;
const BISHOP_MG: i32 = 365;
const ROOK_MG: i32 = 477;
const QUEEN_MG: i32 = 1025;

// Material values (endgame)
const PAWN_EG: i32 = 94;
const KNIGHT_EG: i32 = 281;
const BISHOP_EG: i32 = 297;
const ROOK_EG: i32 = 512;
const QUEEN_EG: i32 = 936;

// PeSTO Piece-Square Tables (from White's perspective, A1=index 0)
// Middlegame tables
const MG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
];

const MG_KNIGHT: [i32; 64] = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
];

const MG_BISHOP: [i32; 64] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
];

const MG_ROOK: [i32; 64] = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
];

const MG_QUEEN: [i32; 64] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9, -19, -30, -15, -13, -22,
];

const MG_KING: [i32; 64] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];

// Endgame tables
const EG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];

const EG_KNIGHT: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -32, -18, -84,  -9,
];

const EG_BISHOP: [i32; 64] = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
];

const EG_ROOK: [i32; 64] = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  12,  12,  11,  11,
      7,   7,   7,   5,   4,  -3,  -5,  -6,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
];

const EG_QUEEN: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
];

const EG_KING: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
];

/// Flip square vertically for black pieces
#[inline(always)]
fn flip(sq: usize) -> usize {
    sq ^ 56
}

/// Get material and PST value for a piece
#[inline(always)]
fn piece_value(piece: Piece, sq: Square, is_white: bool) -> (i32, i32) {
    let idx = if is_white { flip(sq.to_index()) } else { sq.to_index() };
    
    match piece {
        Piece::Pawn => (PAWN_MG + MG_PAWN[idx], PAWN_EG + EG_PAWN[idx]),
        Piece::Knight => (KNIGHT_MG + MG_KNIGHT[idx], KNIGHT_EG + EG_KNIGHT[idx]),
        Piece::Bishop => (BISHOP_MG + MG_BISHOP[idx], BISHOP_EG + EG_BISHOP[idx]),
        Piece::Rook => (ROOK_MG + MG_ROOK[idx], ROOK_EG + EG_ROOK[idx]),
        Piece::Queen => (QUEEN_MG + MG_QUEEN[idx], QUEEN_EG + EG_QUEEN[idx]),
        Piece::King => (MG_KING[idx], EG_KING[idx]),
    }
}

/// Calculate game phase (0 = endgame, 24 = opening)
/// FAST version using bitboard popcounts instead of looping 64 squares
/// Public for use in search optimizations
pub fn game_phase(board: &Board) -> i32 {
    // Use fast bitboard popcount operations
    let knights = board.pieces(Piece::Knight).popcnt() as i32;
    let bishops = board.pieces(Piece::Bishop).popcnt() as i32;
    let rooks = board.pieces(Piece::Rook).popcnt() as i32;
    let queens = board.pieces(Piece::Queen).popcnt() as i32;
    
    let phase = knights * 1 + bishops * 1 + rooks * 2 + queens * 4;
    phase.min(24)
}

/// Evaluate pawn structure
/// Returns (white_score, black_score)
fn evaluate_pawn_structure(board: &Board) -> (i32, i32) {
    let mut white_score = 0;
    let mut black_score = 0;
    
    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    
    // File masks for pawn evaluation
    let file_masks: [u64; 8] = [
        0x0101010101010101, // A file
        0x0202020202020202, // B file
        0x0404040404040404, // C file
        0x0808080808080808, // D file
        0x1010101010101010, // E file
        0x2020202020202020, // F file
        0x4040404040404040, // G file
        0x8080808080808080, // H file
    ];
    
    for sq in ALL_SQUARES {
        if let Some(Piece::Pawn) = board.piece_on(sq) {
            let color = board.color_on(sq).unwrap();
            let file = sq.get_file().to_index();
            let rank = sq.get_rank().to_index();
            let is_white = color == Color::White;
            
            // Passed pawn: no enemy pawns ahead on same or adjacent files
            let ahead_mask = if is_white {
                !0u64 << ((rank + 1) * 8) // All squares ahead for white
            } else {
                !0u64 >> ((8 - rank) * 8) // All squares ahead for black
            };
            
            let adjacent_files = if file == 0 { 
                file_masks[0] | file_masks[1] 
            } else if file == 7 { 
                file_masks[6] | file_masks[7] 
            } else { 
                file_masks[file - 1] | file_masks[file] | file_masks[file + 1] 
            };
            
            let enemy_pawns = if is_white { black_pawns.0 } else { white_pawns.0 };
            let is_passed = (enemy_pawns & ahead_mask & adjacent_files) == 0;
            
            if is_passed {
                // Passed pawn bonus scales with rank (more advanced = more valuable)
                let bonus = if is_white {
                    [0, 10, 20, 40, 60, 100, 150, 0][rank]
                } else {
                    [0, 150, 100, 60, 40, 20, 10, 0][rank]
                };
                if is_white { white_score += bonus; } else { black_score += bonus; }
            }
            
            // Doubled pawns: more than one pawn on same file
            let same_file = file_masks[file];
            let friendly_pawns = if is_white { white_pawns.0 } else { black_pawns.0 };
            if (friendly_pawns & same_file).count_ones() > 1 {
                if is_white { white_score -= 15; } else { black_score -= 15; }
            }
            
            // Isolated pawn: no friendly pawns on adjacent files
            let adj_files = if file == 0 { file_masks[1] } 
                           else if file == 7 { file_masks[6] }
                           else { file_masks[file - 1] | file_masks[file + 1] };
            if (friendly_pawns & adj_files) == 0 {
                if is_white { white_score -= 20; } else { black_score -= 20; }
            }
        }
    }
    
    (white_score, black_score)
}

/// Evaluate king safety
fn evaluate_king_safety(board: &Board) -> (i32, i32) {
    let mut white_score = 0;
    let mut black_score = 0;
    let phase = game_phase(board);
    
    // Only evaluate king safety in middlegame
    if phase < 8 { return (0, 0); }
    
    let white_king = board.king_square(Color::White);
    let black_king = board.king_square(Color::Black);
    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    
    // Pawn shield for white king
    let wk_file = white_king.get_file().to_index();
    let wk_rank = white_king.get_rank().to_index();
    
    // Check if king is castled (on g1/c1 area)
    if wk_rank == 0 && (wk_file <= 2 || wk_file >= 5) {
        // Count pawns in front of king
        for df in -1i32..=1 {
            let f = (wk_file as i32 + df).clamp(0, 7) as usize;
            for dr in 1..=2 {
                let r = wk_rank + dr;
                if r < 8 {
                    let sq = Square::make_square(
                        chess::Rank::from_index(r),
                        chess::File::from_index(f)
                    );
                    if white_pawns.0 & (1u64 << sq.to_index()) != 0 {
                        white_score += 15; // Pawn shield bonus
                    }
                }
            }
        }
    } else if wk_rank > 1 && phase > 12 {
        // King in center during middlegame = bad
        white_score -= 30;
    }
    
    // Same for black king (mirrored)
    let bk_file = black_king.get_file().to_index();
    let bk_rank = black_king.get_rank().to_index();
    
    if bk_rank == 7 && (bk_file <= 2 || bk_file >= 5) {
        for df in -1i32..=1 {
            let f = (bk_file as i32 + df).clamp(0, 7) as usize;
            for dr in 1..=2 {
                if bk_rank >= dr {
                    let r = bk_rank - dr;
                    let sq = Square::make_square(
                        chess::Rank::from_index(r),
                        chess::File::from_index(f)
                    );
                    if black_pawns.0 & (1u64 << sq.to_index()) != 0 {
                        black_score += 15;
                    }
                }
            }
        }
    } else if bk_rank < 6 && phase > 12 {
        black_score -= 30;
    }
    
    (white_score, black_score)
}

/// Evaluate piece coordination (bishop pair, rook on open file)
fn evaluate_pieces(board: &Board) -> (i32, i32) {
    let mut white_score = 0;
    let mut black_score = 0;
    
    let white_bishops = (board.pieces(Piece::Bishop) & board.color_combined(Color::White)).0.count_ones();
    let black_bishops = (board.pieces(Piece::Bishop) & board.color_combined(Color::Black)).0.count_ones();
    
    // Bishop pair bonus
    if white_bishops >= 2 { white_score += 50; }
    if black_bishops >= 2 { black_score += 50; }
    
    // Rook on open file
    let all_pawns = board.pieces(Piece::Pawn).0;
    let file_masks: [u64; 8] = [
        0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    ];
    
    for sq in ALL_SQUARES {
        if let Some(Piece::Rook) = board.piece_on(sq) {
            let color = board.color_on(sq).unwrap();
            let file = sq.get_file().to_index();
            let rank = sq.get_rank().to_index();
            
            // Open file (no pawns)
            if (all_pawns & file_masks[file]) == 0 {
                if color == Color::White { 
                    white_score += 25; 
                } else { 
                    black_score += 25; 
                }
            }
            
            // Rook on 7th rank
            if (color == Color::White && rank == 6) || (color == Color::Black && rank == 1) {
                if color == Color::White {
                    white_score += 40;
                } else {
                    black_score += 40;
                }
            }
        }
    }
    
    (white_score, black_score)
}

/// Evaluate a position using tapered PeSTO evaluation + positional features
/// Returns score in centipawns from White's perspective
#[inline]
pub fn evaluate(board: &Board) -> i32 {
    let mut mg_score = 0i32;
    let mut eg_score = 0i32;
    
    // Iterate only over OCCUPIED squares using bitboards (much faster!)
    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);
    
    // Process each piece type
    for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        let piece_bb = board.pieces(piece);
        
        // White pieces
        for sq in piece_bb & white {
            let (mg, eg) = piece_value(piece, sq, true);
            mg_score += mg;
            eg_score += eg;
        }
        
        // Black pieces  
        for sq in piece_bb & black {
            let (mg, eg) = piece_value(piece, sq, false);
            mg_score -= mg;
            eg_score -= eg;
        }
    }
    
    // Tapered evaluation
    let phase = game_phase(board);
    let mut score = (mg_score * phase + eg_score * (24 - phase)) / 24;
    
    // Fast positional bonuses (no extra loops)
    // Bishop pair bonus
    let white_bishops = (board.pieces(Piece::Bishop) & white).popcnt();
    let black_bishops = (board.pieces(Piece::Bishop) & black).popcnt();
    if white_bishops >= 2 { score += 50; }
    if black_bishops >= 2 { score -= 50; }
    
    // Return from side-to-move perspective
    if board.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

/// Check if a move is a capture
#[inline(always)]
pub fn is_capture(board: &Board, mv: chess::ChessMove) -> bool {
    board.piece_on(mv.get_dest()).is_some()
}

/// MVV-LVA score for move ordering
#[inline(always)]
pub fn mvv_lva_score(board: &Board, mv: chess::ChessMove) -> i32 {
    if let Some(victim) = board.piece_on(mv.get_dest()) {
        let victim_val = match victim {
            Piece::Pawn => 100,
            Piece::Knight => 300,
            Piece::Bishop => 300,
            Piece::Rook => 500,
            Piece::Queen => 900,
            Piece::King => 0,
        };
        
        let attacker = board.piece_on(mv.get_source()).unwrap();
        let attacker_val = match attacker {
            Piece::Pawn => 100,
            Piece::Knight => 300,
            Piece::Bishop => 300,
            Piece::Rook => 500,
            Piece::Queen => 900,
            Piece::King => 0,
        };
        
        // MVV-LVA: prioritize capturing high value with low value
        victim_val * 10 - attacker_val + 10000
    } else {
        0
    }
}

/// Delta evaluation: Calculate score change for a move WITHOUT iterating all 64 squares
/// This is ~10-20x faster than calling evaluate() on the new board
/// Returns the score delta from the moving side's perspective
pub fn evaluate_delta(board: &Board, mv: chess::ChessMove) -> i32 {
    let from = mv.get_source();
    let to = mv.get_dest();
    let piece = board.piece_on(from).unwrap();
    let is_white = board.side_to_move() == Color::White;
    
    let mut delta = 0i32;
    
    // Get current game phase for tapered eval
    let phase = game_phase(board);
    
    // Remove piece from old square, add to new square
    let (old_mg, old_eg) = piece_value(piece, from, is_white);
    let (new_mg, new_eg) = piece_value(piece, to, is_white);
    
    let move_delta_mg = new_mg - old_mg;
    let move_delta_eg = new_eg - old_eg;
    delta += (move_delta_mg * phase + move_delta_eg * (24 - phase)) / 24;
    
    // Handle capture: add captured piece value (we're gaining material)
    if let Some(captured) = board.piece_on(to) {
        let (cap_mg, cap_eg) = piece_value(captured, to, !is_white);
        let cap_value = (cap_mg * phase + cap_eg * (24 - phase)) / 24;
        delta += cap_value; // We capture, so we gain
    }
    
    // Handle promotion
    if let Some(promo) = mv.get_promotion() {
        // Subtract pawn value, add promoted piece value
        let (pawn_mg, pawn_eg) = piece_value(Piece::Pawn, to, is_white);
        let (promo_mg, promo_eg) = piece_value(promo, to, is_white);
        let promo_delta_mg = promo_mg - pawn_mg;
        let promo_delta_eg = promo_eg - pawn_eg;
        delta += (promo_delta_mg * phase + promo_delta_eg * (24 - phase)) / 24;
    }
    
    delta
}

/// Fast static evaluation using delta from previous position
/// prev_eval is the evaluation before the move was made
/// mv is the move that was played
/// board is the board BEFORE the move
pub fn evaluate_after_move(board: &Board, mv: chess::ChessMove, prev_eval: i32) -> i32 {
    // Previous eval is from opponent's perspective (they were to move)
    // After our move, we negate and add delta  
    let delta = evaluate_delta(board, mv);
    -(prev_eval) + delta
}
