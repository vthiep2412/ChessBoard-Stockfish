//! PeSTO Piece-Square Tables and Evaluation
//! Ported from the Python implementation

use chess::{Board, Color, Piece, Square, ALL_SQUARES, ChessMove};

// ============================================================================
// INCREMENTAL EVALUATION STATE
// Tracks MG/EG scores and game phase, updated incrementally on each move
// ============================================================================

/// Incremental evaluation state - tracks scores through search
#[derive(Clone, Copy, Default, Debug)]
pub struct EvalState {
    pub mg_score: i32,  // Middlegame material + PST (from White's perspective)
    pub eg_score: i32,  // Endgame material + PST (from White's perspective)
    pub phase: i32,     // Game phase (0-24)
}

impl EvalState {
    /// Create initial eval state from board
    pub fn new(board: &Board) -> Self {
        let mut mg = 0i32;
        let mut eg = 0i32;
        
        let white = board.color_combined(Color::White);
        let black = board.color_combined(Color::Black);
        
        // Calculate MG/EG scores for all pieces
        for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
            let piece_bb = board.pieces(piece);
            
            for sq in piece_bb & white {
                let (pmg, peg) = piece_value_raw(piece, sq, true);
                mg += pmg;
                eg += peg;
            }
            
            for sq in piece_bb & black {
                let (pmg, peg) = piece_value_raw(piece, sq, false);
                mg -= pmg;
                eg -= peg;
            }
        }
        
        // Calculate phase
        let knights = board.pieces(Piece::Knight).popcnt() as i32;
        let bishops = board.pieces(Piece::Bishop).popcnt() as i32;
        let rooks = board.pieces(Piece::Rook).popcnt() as i32;
        let queens = board.pieces(Piece::Queen).popcnt() as i32;
        let phase = (knights + bishops + rooks * 2 + queens * 4).min(24);
        
        Self { mg_score: mg, eg_score: eg, phase }
    }
    
    /// Apply a move to update scores incrementally
    /// MUST be called BEFORE make_move on board!
    #[inline]
    pub fn apply_move(&mut self, board: &Board, mv: ChessMove) {
        let from = mv.get_source();
        let to = mv.get_dest();
        let is_white = board.side_to_move() == Color::White;
        
        // Get the moving piece
        if let Some(piece) = board.piece_on(from) {
            // Remove piece from source square
            let (mg_from, eg_from) = piece_value_raw(piece, from, is_white);
            // Add piece to destination square
            let (mg_to, eg_to) = piece_value_raw(piece, to, is_white);
            
            let delta_mg = mg_to - mg_from;
            let delta_eg = eg_to - eg_from;
            
            if is_white {
                self.mg_score += delta_mg;
                self.eg_score += delta_eg;
            } else {
                self.mg_score -= delta_mg;
                self.eg_score -= delta_eg;
            }
            
            // Handle captures
            if let Some(captured) = board.piece_on(to) {
                let (cap_mg, cap_eg) = piece_value_raw(captured, to, !is_white);
                if is_white {
                    self.mg_score += cap_mg;
                    self.eg_score += cap_eg;
                } else {
                    self.mg_score -= cap_mg;
                    self.eg_score -= cap_eg;
                }
                
                // Update phase for captured piece
                self.phase -= match captured {
                    Piece::Knight | Piece::Bishop => 1,
                    Piece::Rook => 2,
                    Piece::Queen => 4,
                    _ => 0,
                };
                self.phase = self.phase.max(0);
            }
            
            // Handle en passant
            if piece == Piece::Pawn && board.en_passant() == Some(to) {
                // Captured pawn is on a different square
                let cap_sq = if is_white {
                    Square::make_square(chess::Rank::Fifth, to.get_file())
                } else {
                    Square::make_square(chess::Rank::Fourth, to.get_file())
                };
                let (cap_mg, cap_eg) = piece_value_raw(Piece::Pawn, cap_sq, !is_white);
                if is_white {
                    self.mg_score += cap_mg;
                    self.eg_score += cap_eg;
                } else {
                    self.mg_score -= cap_mg;
                    self.eg_score -= cap_eg;
                }
            }
            
            // Handle promotions
            if let Some(promo) = mv.get_promotion() {
                // Remove pawn value, add promoted piece value
                let (pawn_mg, pawn_eg) = piece_value_raw(Piece::Pawn, to, is_white);
                let (promo_mg, promo_eg) = piece_value_raw(promo, to, is_white);
                
                let delta_mg = promo_mg - pawn_mg;
                let delta_eg = promo_eg - pawn_eg;
                
                if is_white {
                    self.mg_score += delta_mg;
                    self.eg_score += delta_eg;
                } else {
                    self.mg_score -= delta_mg;
                    self.eg_score -= delta_eg;
                }
                
                // Update phase for promotion
                self.phase += match promo {
                    Piece::Knight | Piece::Bishop => 1,
                    Piece::Rook => 2,
                    Piece::Queen => 4,
                    _ => 0,
                };
                self.phase = self.phase.min(24);
            }
            
            // Handle castling (king already moved, need to move rook)
            if piece == Piece::King {
                let from_file = from.get_file().to_index();
                let to_file = to.get_file().to_index();
                
                // Kingside castle
                if from_file == 4 && to_file == 6 {
                    let rook_from = Square::make_square(from.get_rank(), chess::File::H);
                    let rook_to = Square::make_square(from.get_rank(), chess::File::F);
                    let (rf_mg, rf_eg) = piece_value_raw(Piece::Rook, rook_from, is_white);
                    let (rt_mg, rt_eg) = piece_value_raw(Piece::Rook, rook_to, is_white);
                    
                    if is_white {
                        self.mg_score += rt_mg - rf_mg;
                        self.eg_score += rt_eg - rf_eg;
                    } else {
                        self.mg_score -= rt_mg - rf_mg;
                        self.eg_score -= rt_eg - rf_eg;
                    }
                }
                // Queenside castle
                else if from_file == 4 && to_file == 2 {
                    let rook_from = Square::make_square(from.get_rank(), chess::File::A);
                    let rook_to = Square::make_square(from.get_rank(), chess::File::D);
                    let (rf_mg, rf_eg) = piece_value_raw(Piece::Rook, rook_from, is_white);
                    let (rt_mg, rt_eg) = piece_value_raw(Piece::Rook, rook_to, is_white);
                    
                    if is_white {
                        self.mg_score += rt_mg - rf_mg;
                        self.eg_score += rt_eg - rf_eg;
                    } else {
                        self.mg_score -= rt_mg - rf_mg;
                        self.eg_score -= rt_eg - rf_eg;
                    }
                }
            }
        }
    }
    
    /// Get tapered score from side-to-move perspective
    #[inline]
    pub fn score(&self, side_to_move: Color) -> i32 {
        let raw = (self.mg_score * self.phase + self.eg_score * (24 - self.phase)) / 24;
        if side_to_move == Color::White { raw } else { -raw }
    }
}

/// Raw piece value without side-to-move adjustment (for incremental updates)
#[inline]
fn piece_value_raw(piece: Piece, sq: Square, is_white: bool) -> (i32, i32) {
    piece_value(piece, sq, is_white)
}

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
// ============================================================================
// QUADRATIC IMBALANCE TABLES
// Piece values change based on what other pieces are on the board
// Based on Stockfish's approach (simplified)
// ============================================================================

/// Own piece count adjustments for each piece type
/// OWN_IMBALANCE[piece_type][other_piece_type] 
/// e.g., Having 2 bishops (bishop pair) gets a bonus
const OWN_IMBALANCE: [[i32; 5]; 5] = [
    //  Pawn  Knight  Bishop  Rook   Queen
    [   0,      0,      0,      0,      0],  // Pawn
    [   0,     -5,     -2,      0,      0],  // Knight (2 knights = penalty)
    [   3,      5,     25,      0,      0],  // Bishop (bishop pair = big bonus!)
    [  -3,      0,      0,     -5,      0],  // Rook
    [   0,      0,      0,      0,      0],  // Queen
];

/// Opponent piece count adjustments
/// OPP_IMBALANCE[our_piece][opponent_piece]
/// e.g., Knights gain value when opponent has many pawns
const OPP_IMBALANCE: [[i32; 5]; 5] = [
    //  Pawn  Knight  Bishop  Rook   Queen  
    [   0,      0,      0,      0,      0],  // vs Pawn
    [   2,      0,      0,      3,      0],  // Knight vs opponent (good vs pawns, rooks)
    [   0,      3,      0,      2,      0],  // Bishop vs opponent
    [   0,      3,      2,      0,      0],  // Rook vs opponent
    [   0,      0,      0,      0,      0],  // Queen vs opponent
];

/// Evaluate material imbalance
/// Returns score in centipawns (positive = white advantage)
fn evaluate_imbalance(board: &Board) -> i32 {
    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);
    
    // Count pieces for each side
    let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];
    
    let mut white_counts = [0i32; 5];
    let mut black_counts = [0i32; 5];
    
    for (i, &piece) in pieces.iter().enumerate() {
        white_counts[i] = (board.pieces(piece) & white).popcnt() as i32;
        black_counts[i] = (board.pieces(piece) & black).popcnt() as i32;
    }
    
    let mut white_score = 0;
    let mut black_score = 0;
    
    // Calculate imbalance bonuses for each piece type
    for i in 0..5 {
        // Own piece interactions
        for j in 0..5 {
            white_score += white_counts[i] * white_counts[j] * OWN_IMBALANCE[i][j];
            black_score += black_counts[i] * black_counts[j] * OWN_IMBALANCE[i][j];
        }
        
        // Opponent piece interactions
        for j in 0..5 {
            white_score += white_counts[i] * black_counts[j] * OPP_IMBALANCE[i][j];
            black_score += black_counts[i] * white_counts[j] * OPP_IMBALANCE[i][j];
        }
    }
    
    // Divide by 16 to scale (values accumulate quadratically)
    (white_score - black_score) / 16
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

/// Evaluate piece mobility - counts safe squares each piece can move to
/// Returns (white_mobility, black_mobility) in centipawns
/// Based on Stockfish 11 weights: Knight ~4cp/sq, Bishop ~5cp/sq, Rook ~2cp/sq, Queen ~1cp/sq
fn evaluate_mobility(board: &Board) -> (i32, i32) {
    use chess::{get_knight_moves, get_bishop_moves, get_rook_moves, EMPTY, BitBoard};
    
    let mut white_score = 0i32;
    let mut black_score = 0i32;
    
    let white_pieces = board.color_combined(Color::White);
    let black_pieces = board.color_combined(Color::Black);
    let all_pieces = board.combined();
    
    // Squares attacked by enemy pawns are NOT safe
    let white_pawns = board.pieces(Piece::Pawn) & white_pieces;
    let black_pawns = board.pieces(Piece::Pawn) & black_pieces;
    
    // Pawn attack masks (simplified - left and right captures)
    let white_pawn_attacks = ((white_pawns.0 << 7) & 0xfefefefefefefefe) | 
                              ((white_pawns.0 << 9) & 0x7f7f7f7f7f7f7f7f);
    let black_pawn_attacks = ((black_pawns.0 >> 7) & 0x7f7f7f7f7f7f7f7f) | 
                              ((black_pawns.0 >> 9) & 0xfefefefefefefefe);
    
    // Safe squares for each side (not attacked by enemy pawns, not blocked by own pieces)
    let white_safe = !(BitBoard(black_pawn_attacks) | white_pieces);
    let black_safe = !(BitBoard(white_pawn_attacks) | black_pieces);
    
    // Knight mobility (~4 cp per square)
    for sq in board.pieces(Piece::Knight) & white_pieces {
        let attacks = get_knight_moves(sq);
        let safe_moves = (attacks & white_safe).popcnt() as i32;
        white_score += safe_moves * 4;
    }
    for sq in board.pieces(Piece::Knight) & black_pieces {
        let attacks = get_knight_moves(sq);
        let safe_moves = (attacks & black_safe).popcnt() as i32;
        black_score += safe_moves * 4;
    }
    
    // Bishop mobility (~5 cp per square)
    for sq in board.pieces(Piece::Bishop) & white_pieces {
        let attacks = get_bishop_moves(sq, *all_pieces);
        let safe_moves = (attacks & white_safe).popcnt() as i32;
        white_score += safe_moves * 5;
    }
    for sq in board.pieces(Piece::Bishop) & black_pieces {
        let attacks = get_bishop_moves(sq, *all_pieces);
        let safe_moves = (attacks & black_safe).popcnt() as i32;
        black_score += safe_moves * 5;
    }
    
    // Rook mobility (~2 cp per square)
    for sq in board.pieces(Piece::Rook) & white_pieces {
        let attacks = get_rook_moves(sq, *all_pieces);
        let safe_moves = (attacks & white_safe).popcnt() as i32;
        white_score += safe_moves * 2;
    }
    for sq in board.pieces(Piece::Rook) & black_pieces {
        let attacks = get_rook_moves(sq, *all_pieces);
        let safe_moves = (attacks & black_safe).popcnt() as i32;
        black_score += safe_moves * 2;
    }
    
    // Queen mobility (~1 cp per square, less weight as queens are naturally mobile)
    for sq in board.pieces(Piece::Queen) & white_pieces {
        let attacks = get_bishop_moves(sq, *all_pieces) | get_rook_moves(sq, *all_pieces);
        let safe_moves = (attacks & white_safe).popcnt() as i32;
        white_score += safe_moves * 1;
    }
    for sq in board.pieces(Piece::Queen) & black_pieces {
        let attacks = get_bishop_moves(sq, *all_pieces) | get_rook_moves(sq, *all_pieces);
        let safe_moves = (attacks & black_safe).popcnt() as i32;
        black_score += safe_moves * 1;
    }
    
    (white_score, black_score)
}

/// Lazy evaluation margin - skip expensive terms if score far from window
const LAZY_MARGIN: i32 = 300; // ~3 pawns - tune this!

/// Fast evaluation using only PeSTO + bishop pair (no expensive loops)
/// Used for lazy eval cutoffs
#[inline]
pub fn fast_evaluate(board: &Board) -> i32 {
    let mut mg_score = 0i32;
    let mut eg_score = 0i32;
    
    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);
    
    // Process each piece type
    for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        let piece_bb = board.pieces(piece);
        
        for sq in piece_bb & white {
            let (mg, eg) = piece_value(piece, sq, true);
            mg_score += mg;
            eg_score += eg;
        }
        
        for sq in piece_bb & black {
            let (mg, eg) = piece_value(piece, sq, false);
            mg_score -= mg;
            eg_score -= eg;
        }
    }
    
    // Tapered evaluation
    let phase = game_phase(board);
    let mut score = (mg_score * phase + eg_score * (24 - phase)) / 24;
    
    // Quadratic imbalance (includes bishop pair!)
    score += evaluate_imbalance(board);
    
    // Return from side-to-move perspective
    if board.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

/// Helper to get piece value for SEE
fn piece_val_see(p: Piece) -> i32 {
    match p {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}

/// Helper to get attackers with custom occupied bitboard
/// Used for Static Exchange Evaluation (SEE)
fn get_attackers_to(sq: Square, occupied: chess::BitBoard, board: &Board) -> chess::BitBoard {
    use chess::{get_bishop_moves, get_rook_moves, get_knight_moves, get_king_moves, get_pawn_attacks};

    let mut attackers = chess::BitBoard(0);

    // Knights
    attackers |= get_knight_moves(sq) & board.pieces(Piece::Knight);

    // King
    attackers |= get_king_moves(sq) & board.pieces(Piece::King);

    // Pawns
    // Attackers from White Pawns = Squares that a Black Pawn at `sq` would attack (reverse capture)
    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);

    // Note: get_pawn_attacks(sq, color, include_defenders)
    attackers |= get_pawn_attacks(sq, Color::Black, chess::BitBoard(0)) & white_pawns;
    attackers |= get_pawn_attacks(sq, Color::White, chess::BitBoard(0)) & black_pawns;

    // Sliding pieces (Bishop/Queen diagonal)
    // We must recalculate these with the new 'occupied' bitboard
    let diag_sliders = board.pieces(Piece::Bishop) | board.pieces(Piece::Queen);
    if diag_sliders.0 != 0 {
        attackers |= get_bishop_moves(sq, occupied) & diag_sliders;
    }

    // Sliding pieces (Rook/Queen orthogonal)
    let orth_sliders = board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    if orth_sliders.0 != 0 {
        attackers |= get_rook_moves(sq, occupied) & orth_sliders;
    }

    attackers
}

/// Static Exchange Evaluation (SEE)
/// Calculates the approximate material outcome of a capture sequence on a square
/// Used for pruning bad captures in search
pub fn see(board: &Board, mv: ChessMove) -> i32 {
    let from = mv.get_source();
    let to = mv.get_dest();

    // Initial gain
    let mut gain = [0i32; 32];
    let mut d = 0;

    let mut attacker = board.piece_on(from).unwrap();

    // Value of the piece being captured
    let mut victim_val = if let Some(p) = board.piece_on(to) {
        piece_val_see(p)
    } else if board.en_passant() == Some(to) && attacker == Piece::Pawn {
        100
    } else {
        0
    };

    // Handle promotion on the first move
    if let Some(promo) = mv.get_promotion() {
        gain[d] = victim_val + piece_val_see(promo) - 100; // Gain promo value, lose pawn value
        attacker = promo;
    } else {
        gain[d] = victim_val;
    }
    d += 1;

    // If not a capture and no promotion, SEE is 0
    if gain[0] == 0 && mv.get_promotion().is_none() {
        return 0;
    }

    // Mask out the moving piece to simulate the move
    // Note: chess::BitBoard wraps u64, so we use logic operators on it
    let mut occupied = *board.combined() ^ chess::BitBoard::from_square(from);
    let mut side = !board.side_to_move();

    // Iterative exchange
    loop {
        // Find attackers for the current side
        let attackers = get_attackers_to(to, occupied, board) & board.color_combined(side);

        if attackers.0 == 0 { break; }

        // Find Least Valuable Attacker (LVA)
        // Order: Pawn, Knight, Bishop, Rook, Queen, King
        let mut next_att_sq = None;
        let mut next_att_piece = Piece::King;

        for p in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
             let piece_attackers = attackers & board.pieces(p);
             if piece_attackers.0 != 0 {
                 next_att_piece = p;
                 // Get specific square (LSB)
                 next_att_sq = Some(piece_attackers.to_square());
                 break;
             }
        }

        if let Some(sq) = next_att_sq {
            // Remove the attacker from occupied (make the capture)
            occupied ^= chess::BitBoard::from_square(sq);

            // Gain for this side is: Value of piece captured - Gain of previous side
            // Note: gain array stores cumulative balance from POV of side that started
            gain[d] = piece_val_see(attacker) - gain[d-1];

            attacker = next_att_piece;
            side = !side;
            d += 1;
        } else {
            break;
        }
    }

    // Minimax the gain array back up to find true value
    // Each side will choose to stop if capturing leads to a worse result
    while d > 1 {
        d -= 1;
        gain[d-1] = -(-gain[d-1]).max(gain[d]);
    }

    gain[0]
}

/// Evaluate space (central control)
/// Bonus for controlling squares in enemy territory/center
fn evaluate_space(board: &Board) -> (i32, i32) {
    let phase = game_phase(board);

    // Space is mostly important in middlegame
    if phase < 10 { return (0, 0); }

    let mut white_score = 0;
    let mut black_score = 0;

    let white_occ = board.color_combined(Color::White);
    let black_occ = board.color_combined(Color::Black);

    // Safe squares center mask (files C-F, ranks 2-5 for white, 3-6 for black)
    // 0x00003C3C3C3C0000 is center 4x4
    let center_mask = 0x00003C3C3C3C0000;

    // White space: squares attacked by white non-pawns in center/enemy half
    // that are NOT attacked by black pawns
    // Simplified: Just bonus for piece presence in center

    let safe_mask_white = !0; // TODO: Filter by black pawn attacks
    let safe_mask_black = !0; // TODO: Filter by white pawn attacks

    // Simple bonus for pieces in center
    let white_center = (white_occ.0 & center_mask).count_ones() as i32;
    let black_center = (black_occ.0 & center_mask).count_ones() as i32;

    white_score += white_center * 10;
    black_score += black_center * 10;

    (white_score, black_score)
}

/// Evaluate outposts (Knights/Bishops on strong squares)
fn evaluate_outposts(board: &Board) -> (i32, i32) {
    let mut white_score = 0;
    let mut black_score = 0;

    let white_pieces = board.color_combined(Color::White);
    let black_pieces = board.color_combined(Color::Black);
    let white_pawns = board.pieces(Piece::Pawn) & white_pieces;
    let black_pawns = board.pieces(Piece::Pawn) & black_pieces;

    // White outposts: Ranks 4, 5, 6
    // Supported by white pawn
    // Not attackable by black pawns (simplified: just supported for now)

    for piece in [Piece::Knight, Piece::Bishop] {
        let w_pieces = board.pieces(piece) & white_pieces;
        let b_pieces = board.pieces(piece) & black_pieces;

        for sq in w_pieces {
            let rank = sq.get_rank().to_index();
            if rank >= 3 && rank <= 5 {
                // Check support by pawn (White Piece supported by White Pawn)
                // Reverse: Black Pawn at sq attacks where White Pawn must be
                let support_sqs = chess::get_pawn_attacks(sq, Color::Black, chess::BitBoard(0));
                if (support_sqs & white_pawns).0 != 0 {
                    white_score += 25; // Supported outpost

                    // Bonus if on opponent half
                    if rank >= 4 { white_score += 15; }
                }
            }
        }

        for sq in b_pieces {
            let rank = sq.get_rank().to_index();
            if rank >= 2 && rank <= 4 { // Black perspective ranks 5, 4, 3
                // Check support by pawn (Black Piece supported by Black Pawn)
                // Reverse: White Pawn at sq attacks where Black Pawn must be
                let support_sqs = chess::get_pawn_attacks(sq, Color::White, chess::BitBoard(0));
                if (support_sqs & black_pawns).0 != 0 {
                    black_score += 25;
                    if rank <= 3 { black_score += 15; }
                }
            }
        }
    }

    (white_score, black_score)
}

/// Full evaluation with all positional terms
/// Returns score from side-to-move perspective
#[inline]
fn full_evaluate(board: &Board, base_score: i32) -> i32 {
    let mut score = base_score;
    
    // Expensive terms - only compute when needed
    let (white_pawn, black_pawn) = evaluate_pawn_structure(board);
    let (white_king, black_king) = evaluate_king_safety(board);
    let (white_mobility, black_mobility) = evaluate_mobility(board);
    let (white_space, black_space) = evaluate_space(board);
    let (white_outpost, black_outpost) = evaluate_outposts(board);
    
    let positional = (white_pawn - black_pawn) + (white_king - black_king) +
                     (white_mobility - black_mobility) + (white_space - black_space) +
                     (white_outpost - black_outpost);
    
    // Adjust score based on side to move
    let mut final_score = if board.side_to_move() == Color::White {
        score + positional
    } else {
        score - positional
    };
    
    // ==========================================
    // TEMPO BONUS - Reward having the move
    // This encourages the engine to prefer positions where it has initiative
    // ==========================================
    final_score += 12; // Small tempo bonus for side to move
    
    // ==========================================
    // CONTEMPT FACTOR - Avoid draws when ahead
    // When we have material advantage, penalize drawish positions
    // This makes the engine fight for wins instead of accepting draws
    // ==========================================
    let phase = game_phase(board);
    
    // If we're ahead, add contempt to avoid draws
    // Contempt is stronger in middlegame (more chances to convert)
    if final_score > 50 {
        // We're winning - add contempt bonus to avoid lazy play
        let contempt = (phase * 5) / 24; // 0-5 cp based on phase
        final_score += contempt;
    } else if final_score < -50 {
        // We're losing - add contempt to fight harder (negative for opponent)
        let contempt = (phase * 5) / 24;
        final_score -= contempt;
    }
    
    final_score
}

/// Evaluate a position using tapered PeSTO evaluation + positional features
/// Uses lazy evaluation: skips expensive terms if score far from alpha-beta window
/// Returns score in centipawns from side-to-move perspective
#[inline]
pub fn evaluate(board: &Board) -> i32 {
    // Fast path: compute base score
    let base_score = fast_evaluate(board);
    
    // Full evaluation (no lazy cutoff when called without alpha/beta)
    full_evaluate(board, base_score)
}

/// Lazy evaluation: skip expensive terms if score far from alpha-beta window
/// This is the optimized version called from search with alpha/beta bounds
#[inline]
pub fn evaluate_lazy(board: &Board, alpha: i32, beta: i32) -> i32 {
    // Fast path: compute base score
    let base_score = fast_evaluate(board);
    
    // Lazy cutoff: if score is way outside window, skip expensive terms
    if base_score + LAZY_MARGIN <= alpha {
        return base_score; // Failing low - skip expensive eval
    }
    if base_score - LAZY_MARGIN >= beta {
        return base_score; // Failing high - skip expensive eval  
    }
    
    // Score is close to window - compute full evaluation
    full_evaluate(board, base_score)
}

/// Incremental evaluation using pre-computed EvalState
/// Uses lazy cutoffs like evaluate_lazy but with pre-computed base score
#[inline]
pub fn evaluate_with_state(board: &Board, state: &EvalState, alpha: i32, beta: i32) -> i32 {
    // Use pre-computed base score from EvalState + imbalance
    let base_score = state.score(board.side_to_move()) + evaluate_imbalance(board);
    
    // Lazy cutoff
    if base_score + LAZY_MARGIN <= alpha {
        return base_score;
    }
    if base_score - LAZY_MARGIN >= beta {
        return base_score;
    }
    
    // Compute full eval with expensive terms
    full_evaluate(board, base_score)
}

/// Check if a move is a capture (includes en passant!)
#[inline(always)]
pub fn is_capture(board: &Board, mv: chess::ChessMove) -> bool {
    // Normal capture: piece on destination
    if board.piece_on(mv.get_dest()).is_some() {
        return true;
    }
    // En passant: pawn moves diagonally to empty square (en passant target)
    if let Some(ep_sq) = board.en_passant() {
        if board.piece_on(mv.get_source()) == Some(Piece::Pawn) 
           && mv.get_dest() == ep_sq {
            return true;
        }
    }
    false
}

/// MVV-LVA score for move ordering (handles en passant!)
#[inline(always)]
pub fn mvv_lva_score(board: &Board, mv: chess::ChessMove) -> i32 {
    // Determine victim piece value
    let victim_val = if let Some(victim) = board.piece_on(mv.get_dest()) {
        match victim {
            Piece::Pawn => 100,
            Piece::Knight => 300,
            Piece::Bishop => 300,
            Piece::Rook => 500,
            Piece::Queen => 900,
            Piece::King => 0,
        }
    } else {
        // Check for en passant: pawn captures to en passant square
        if let Some(ep_sq) = board.en_passant() {
            if board.piece_on(mv.get_source()) == Some(Piece::Pawn) && mv.get_dest() == ep_sq {
                100 // Capturing a pawn via en passant
            } else {
                return 0; // Not a capture
            }
        } else {
            return 0; // Not a capture
        }
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
    } else if piece == Piece::Pawn {
        // CRITICAL: Handle en passant capture!
        // En passant: pawn moves diagonally to empty square, captures pawn beside it
        if let Some(ep_sq) = board.en_passant() {
            if to == ep_sq {
                // Captured pawn is on a different rank than destination
                // White captures: pawn is on rank 5 (ep square is rank 6)  
                // Black captures: pawn is on rank 4 (ep square is rank 3)
                let captured_sq = if is_white {
                    Square::make_square(chess::Rank::Fifth, to.get_file())
                } else {
                    Square::make_square(chess::Rank::Fourth, to.get_file())
                };
                let (cap_mg, cap_eg) = piece_value(Piece::Pawn, captured_sq, !is_white);
                let cap_value = (cap_mg * phase + cap_eg * (24 - phase)) / 24;
                delta += cap_value;
            }
        }
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
    
    // Handle castling rook movement
    if piece == Piece::King {
        let from_file = from.get_file().to_index();
        let to_file = to.get_file().to_index();

        // Check if move is castling (King moves 2 squares)
        if (from_file as i32 - to_file as i32).abs() == 2 {
            let rank = from.get_rank();
            // Determine rook source and dest
            // Kingside: file 4 -> 6 (e -> g). Rook: h -> f (file 7 -> 5)
            // Queenside: file 4 -> 2 (e -> c). Rook: a -> d (file 0 -> 3)

            let (rook_from_file, rook_to_file) = if to_file > from_file {
                // Kingside
                (chess::File::H, chess::File::F)
            } else {
                // Queenside
                (chess::File::A, chess::File::D)
            };

            let rook_from = Square::make_square(rank, rook_from_file);
            let rook_to = Square::make_square(rank, rook_to_file);

            let (r_old_mg, r_old_eg) = piece_value(Piece::Rook, rook_from, is_white);
            let (r_new_mg, r_new_eg) = piece_value(Piece::Rook, rook_to, is_white);

            let r_delta_mg = r_new_mg - r_old_mg;
            let r_delta_eg = r_new_eg - r_old_eg;

            delta += (r_delta_mg * phase + r_delta_eg * (24 - phase)) / 24;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use chess::{Board, ChessMove, Square, Rank, File, Piece};
    use std::str::FromStr;

    #[test]
    fn test_castling_delta_includes_rook() {
        // Position where white can castle kingside
        let fen = "r3k2r/pppq1ppp/2npb3/2b1p3/2B1P3/2NPB3/PPPQ1PPP/R3K2R w KQkq - 4 8";
        let board = Board::from_str(fen).unwrap();

        // e1g1 (White Kingside Castle)
        let from = Square::make_square(Rank::First, File::E);
        let to = Square::make_square(Rank::First, File::G);
        let m = ChessMove::new(from, to, None);

        let delta = evaluate_delta(&board, m);

        // Calculate expected components
        let phase = game_phase(&board);
        let is_white = true; // White to move

        // King
        let (k_old_mg, k_old_eg) = piece_value(Piece::King, from, is_white);
        let (k_new_mg, k_new_eg) = piece_value(Piece::King, to, is_white);
        let k_delta_mg = k_new_mg - k_old_mg;
        let k_delta_eg = k_new_eg - k_old_eg;
        let k_val = (k_delta_mg * phase + k_delta_eg * (24 - phase)) / 24;

        // Rook (h1 -> f1)
        let r_from = Square::make_square(Rank::First, File::H);
        let r_to = Square::make_square(Rank::First, File::F);
        let (r_old_mg, r_old_eg) = piece_value(Piece::Rook, r_from, is_white);
        let (r_new_mg, r_new_eg) = piece_value(Piece::Rook, r_to, is_white);
        let r_delta_mg = r_new_mg - r_old_mg;
        let r_delta_eg = r_new_eg - r_old_eg;
        let r_val = (r_delta_mg * phase + r_delta_eg * (24 - phase)) / 24;

        let expected_delta = k_val + r_val;

        println!("Phase: {}", phase);
        println!("King delta: {}", k_val);
        println!("Rook delta: {}", r_val);
        println!("Actual delta: {}", delta);
        println!("Expected delta: {}", expected_delta);

        assert_eq!(delta, expected_delta, "Evaluate delta should include rook movement for castling");
    }

    #[test]
    fn test_castling_delta_queenside_black() {
        // Position where black can castle queenside
        // r3k2r/pppq1ppp/2npb3/2b1p3/2B1P3/2NPB3/PPPQ1PPP/R3K2R b KQkq - 4 8
        let fen = "r3k2r/pppq1ppp/2npb3/2b1p3/2B1P3/2NPB3/PPPQ1PPP/R3K2R b KQkq - 4 8";
        let board = Board::from_str(fen).unwrap();

        // e8c8 (Black Queenside Castle)
        let from = Square::make_square(Rank::Eighth, File::E);
        let to = Square::make_square(Rank::Eighth, File::C);
        let m = ChessMove::new(from, to, None);

        let delta = evaluate_delta(&board, m);

        let phase = game_phase(&board);
        let is_white = false; // Black to move

        // King (e8 -> c8)
        let (k_old_mg, k_old_eg) = piece_value(Piece::King, from, is_white);
        let (k_new_mg, k_new_eg) = piece_value(Piece::King, to, is_white);
        let k_delta_mg = k_new_mg - k_old_mg;
        let k_delta_eg = k_new_eg - k_old_eg;
        let k_val = (k_delta_mg * phase + k_delta_eg * (24 - phase)) / 24;

        // Rook (a8 -> d8)
        let r_from = Square::make_square(Rank::Eighth, File::A);
        let r_to = Square::make_square(Rank::Eighth, File::D);
        let (r_old_mg, r_old_eg) = piece_value(Piece::Rook, r_from, is_white);
        let (r_new_mg, r_new_eg) = piece_value(Piece::Rook, r_to, is_white);
        let r_delta_mg = r_new_mg - r_old_mg;
        let r_delta_eg = r_new_eg - r_old_eg;
        let r_val = (r_delta_mg * phase + r_delta_eg * (24 - phase)) / 24;

        let expected_delta = k_val + r_val;

        println!("Phase: {}", phase);
        println!("King delta: {}", k_val);
        println!("Rook delta: {}", r_val);
        println!("Actual delta: {}", delta);
        println!("Expected delta: {}", expected_delta);

        assert_eq!(delta, expected_delta, "Evaluate delta should include rook movement for black queenside castling");
    }
}
