//! Alpha-Beta Search with Transposition Table, Iterative Deepening, and Parallel Search

use chess::{Board, MoveGen, ChessMove, BoardStatus};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use crate::eval;

// =============================================================================
// TRANSPOSITION TABLE - Fixed-size array for O(1) access
// =============================================================================

/// TT size: 2^22 = 4,194,304 entries (~64MB with 16 bytes per entry)
const TT_SIZE: usize = 1 << 22;
const TT_MASK: usize = TT_SIZE - 1;

/// Compact TT entry - 16 bytes total
#[derive(Clone, Copy)]
#[repr(C)]
struct TTEntry {
    hash: u64,              // Full hash for collision detection
    best_move: u16,         // Encoded move (from 6 bits + to 6 bits + promo 4 bits)
    score: i16,             // Score (clamped to i16 range)
    depth: u8,              // Search depth
    flag: u8,               // 0=None, 1=Exact, 2=Alpha, 3=Beta
    _padding: [u8; 2],      // Alignment padding
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            best_move: 0,
            score: 0,
            depth: 0,
            flag: 0,
            _padding: [0; 2],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
enum TTFlag {
    None = 0,
    Exact = 1,
    Alpha = 2,
    Beta = 3,
}

impl From<u8> for TTFlag {
    fn from(v: u8) -> Self {
        match v {
            1 => TTFlag::Exact,
            2 => TTFlag::Alpha,
            3 => TTFlag::Beta,
            _ => TTFlag::None,
        }
    }
}

/// Encode a ChessMove into u16
#[inline(always)]
fn encode_move(mv: ChessMove) -> u16 {
    let from = mv.get_source().to_index() as u16;
    let to = mv.get_dest().to_index() as u16;
    let promo = match mv.get_promotion() {
        Some(chess::Piece::Knight) => 1,
        Some(chess::Piece::Bishop) => 2,
        Some(chess::Piece::Rook) => 3,
        Some(chess::Piece::Queen) => 4,
        _ => 0,
    };
    (from) | (to << 6) | (promo << 12)
}

/// Decode a u16 into ChessMove (requires board to validate)
/// SAFE: Validates indices to prevent UB with corrupted TT data
#[inline(always)]
fn decode_move(encoded: u16, _board: &Board) -> Option<ChessMove> {
    if encoded == 0 {
        return None;
    }
    
    // Extract and VALIDATE indices to prevent UB with corrupted TT data
    let from_idx = (encoded & 0x3F) as u8;
    let to_idx = ((encoded >> 6) & 0x3F) as u8;
    
    // Bounds check: square indices must be 0-63
    if from_idx >= 64 || to_idx >= 64 {
        return None; // Corrupted TT entry, skip this move
    }
    
    // SAFETY: We just validated indices are in range 0-63
    let from = unsafe { chess::Square::new(from_idx) };
    let to = unsafe { chess::Square::new(to_idx) };
    
    let promo = match (encoded >> 12) & 0xF {
        1 => Some(chess::Piece::Knight),
        2 => Some(chess::Piece::Bishop),
        3 => Some(chess::Piece::Rook),
        4 => Some(chess::Piece::Queen),
        _ => None,
    };
    Some(ChessMove::new(from, to, promo))
}

// =============================================================================
// LOCK-FREE GLOBAL TRANSPOSITION TABLE
// Uses raw pointer access for zero-overhead reads/writes
// =============================================================================

use std::cell::UnsafeCell;
use std::sync::Once;

/// Global TT storage - initialized once on first use
struct GlobalTT {
    data: UnsafeCell<Vec<TTEntry>>,
}

// SAFETY: We accept racy reads/writes - this is standard in chess engines
// Entries are small (16 bytes) and atomic-ish on 64-bit systems
unsafe impl Sync for GlobalTT {}
unsafe impl Send for GlobalTT {}

static mut GLOBAL_TT: Option<GlobalTT> = None;
static INIT: Once = Once::new();

/// Get pointer to global TT (lazy initialization)
#[inline(always)]
fn get_tt() -> *mut TTEntry {
    unsafe {
        INIT.call_once(|| {
            let mut v = Vec::with_capacity(TT_SIZE);
            v.resize(TT_SIZE, TTEntry::default());
            GLOBAL_TT = Some(GlobalTT {
                data: UnsafeCell::new(v),
            });
        });
        (*GLOBAL_TT.as_ref().unwrap().data.get()).as_mut_ptr()
    }
}

/// Probe the transposition table - ZERO OVERHEAD
#[inline(always)]
fn tt_probe(hash: u64) -> Option<(TTEntry, TTFlag)> {
    let idx = (hash as usize) & TT_MASK;
    // SAFETY: idx is always < TT_SIZE due to mask, TT is initialized
    let entry = unsafe { *get_tt().add(idx) };
    if entry.hash == hash && entry.flag != 0 {
        Some((entry, TTFlag::from(entry.flag)))
    } else {
        None
    }
}

/// Store an entry in the transposition table - ZERO OVERHEAD
#[inline(always)]
fn tt_store(hash: u64, depth: u8, score: i32, flag: TTFlag, best_move: Option<ChessMove>) {
    let idx = (hash as usize) & TT_MASK;
    // SAFETY: idx is always < TT_SIZE due to mask, TT is initialized
    unsafe {
        let entry_ptr = get_tt().add(idx);
        let existing = *entry_ptr;
        
        // Depth-preferred replacement: replace if deeper or same position
        if depth >= existing.depth || existing.hash == hash || existing.flag == 0 {
            *entry_ptr = TTEntry {
                hash,
                depth,
                score: score.clamp(-30000, 30000) as i16,
                flag: flag as u8,
                best_move: best_move.map(encode_move).unwrap_or(0),
                _padding: [0; 2],
            };
        }
    }
}

/// Global debug flag - set to true for verbose logging
pub static DEBUG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable debug mode
pub fn set_debug(enabled: bool) {
    DEBUG.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Check if debug mode is enabled
#[inline]
fn is_debug() -> bool {
    DEBUG.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clear the transposition table - CRITICAL for fresh searches!
/// 
/// # Safety
/// **MUST NOT be called during active parallel searches!**
/// Only call from Python/main thread BEFORE starting a search:
/// - At ucinewgame command
/// - At start of benchmark
/// - Never during parallel_root_search or negamax execution
/// 
/// Calling during active search causes data races and undefined behavior!
pub fn tt_clear() {
    unsafe {
        if let Some(tt) = GLOBAL_TT.as_ref() {
            let ptr = (*tt.data.get()).as_mut_ptr();
            // Reset entire TT memory to 0 (zeroed entries have flag=0 = invalid)
            std::ptr::write_bytes(ptr, 0, TT_SIZE);
            if is_debug() {
                eprintln!("[DEBUG] TT cleared - {} entries zeroed", TT_SIZE);
            }
        }
    }
}

/// Killer moves table (indexed by depth)
/// NOTE: thread_local means each thread maintains separate killer moves.
/// This is simpler but means parallel workers don't share killer move knowledge.
/// A shared atomic approach would allow knowledge sharing but adds complexity.
thread_local! {
    static KILLERS: std::cell::RefCell<[[Option<ChessMove>; 2]; 64]> = 
        std::cell::RefCell::new([[None; 2]; 64]);
}

/// History heuristic table [from_sq][to_sq] for move ordering
/// NOTE: thread_local - same trade-off as KILLERS above
thread_local! {
    static HISTORY: std::cell::RefCell<[[i32; 64]; 64]> = 
        std::cell::RefCell::new([[0; 64]; 64]);
}

/// Counter-Move History: indexed by PREVIOUS move's [from][to]
/// Stores history values for moves that worked after specific opponent moves
/// Example: if Nf3 often works after e4, counter_history[e2][e4][g1][f3] is high
thread_local! {
    static COUNTER_HISTORY: std::cell::RefCell<[[[[i16; 64]; 64]; 64]; 64]> = 
        std::cell::RefCell::new([[[[0; 64]; 64]; 64]; 64]);
}

/// Aggressiveness level (1-10) for pruning - set by iterative_deepening
thread_local! {
    static AGGRESSIVENESS: std::cell::Cell<u8> = std::cell::Cell::new(5);
}

const INFINITY: i32 = 30000;
const MATE_SCORE: i32 = 29000;
/// Contempt factor: negative value means engine dislikes draws
/// Higher absolute value = more aggressive, but risk of overextending
const CONTEMPT: i32 = -15; // Slight draw aversion (engine sees draws as bad)

// Node counters for diagnostics - ATOMIC for thread-safe parallel search
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
static NODE_COUNT: AtomicU64 = AtomicU64::new(0);
static QNODE_COUNT: AtomicU64 = AtomicU64::new(0);
static STOP_SEARCH: AtomicBool = AtomicBool::new(false);

/// Time Management
#[derive(Clone, Copy)]
pub struct TimeManager {
    pub start_time: std::time::Instant,
    pub allocated_time: u128, // millis
}

impl TimeManager {
    pub fn new(wtime: Option<u64>, btime: Option<u64>, movestogo: Option<u64>, turn: chess::Color) -> Option<Self> {
        let time_left = if turn == chess::Color::White { wtime } else { btime };
        
        if let Some(t) = time_left {
            let moves = movestogo.unwrap_or(30).max(1);
            let mut alloc = t / moves;
            
            // Safety buffer (don't use all time)
            alloc = alloc.saturating_sub(50);
            
            // Minimum search time
            if alloc < 50 { alloc = 50; }
            
            Some(Self {
                start_time: std::time::Instant::now(),
                allocated_time: alloc as u128,
            })
        } else {
            None
        }
    }
    
    #[inline(always)]
    pub fn check_time(&self) -> bool {
        self.start_time.elapsed().as_millis() > self.allocated_time
    }
}

/// Reset node counters (thread-safe)
pub fn reset_node_counts() {
    NODE_COUNT.store(0, Ordering::Relaxed);
    QNODE_COUNT.store(0, Ordering::Relaxed);
    STOP_SEARCH.store(false, Ordering::Relaxed);
}

/// Get node counts (thread-safe)
pub fn get_node_counts() -> (u64, u64) {
    (NODE_COUNT.load(Ordering::Relaxed), QNODE_COUNT.load(Ordering::Relaxed))
}

/// Update killer moves for a quiet move that caused a beta cutoff
#[inline(always)]
fn update_killers(ply: u8, mv: ChessMove) {
    let ply_idx = (ply as usize).min(63);  // Clamp to valid range
    KILLERS.with(|k| {
        let mut killers = k.borrow_mut();
        if killers[ply_idx][0] != Some(mv) {
            killers[ply_idx][1] = killers[ply_idx][0];
            killers[ply_idx][0] = Some(mv);
        }
    });
}

/// Update history heuristic for a quiet move that caused a beta cutoff
#[inline(always)]
fn update_history(mv: ChessMove, depth: u8) {
    let from = mv.get_source().to_index();
    let to = mv.get_dest().to_index();
    HISTORY.with(|h| {
        let mut hist = h.borrow_mut();
        hist[from][to] += (depth as i32) * (depth as i32);
        if hist[from][to] > 10000 {
            hist[from][to] = 10000;
        }
    });
}

/// Update counter-move history: what worked after a specific previous move
#[inline(always)]
fn update_counter_history(prev_mv: Option<ChessMove>, curr_mv: ChessMove, depth: u8) {
    if let Some(prev) = prev_mv {
        let pf = prev.get_source().to_index();
        let pt = prev.get_dest().to_index();
        let cf = curr_mv.get_source().to_index();
        let ct = curr_mv.get_dest().to_index();
        
        COUNTER_HISTORY.with(|ch| {
            let mut hist = ch.borrow_mut();
            let bonus = (depth as i16) * (depth as i16);
            hist[pf][pt][cf][ct] = hist[pf][pt][cf][ct].saturating_add(bonus).min(10000);
        });
    }
}

/// Get counter-history score for a move based on previous move
#[inline(always)]
fn get_counter_history(prev_mv: Option<ChessMove>, curr_mv: ChessMove) -> i32 {
    if let Some(prev) = prev_mv {
        let pf = prev.get_source().to_index();
        let pt = prev.get_dest().to_index();
        let cf = curr_mv.get_source().to_index();
        let ct = curr_mv.get_dest().to_index();
        
        COUNTER_HISTORY.with(|ch| {
            ch.borrow()[pf][pt][cf][ct] as i32
        })
    } else {
        0
    }
}


/// Get futility margin based on aggressiveness (higher aggr = wider margin = more pruning)
fn get_futility_margin(depth: u8) -> i32 {
    let aggr = AGGRESSIVENESS.with(|a| a.get()) as i32;
    let base = [0, 150, 300, 500][depth.min(3) as usize];
    base + aggr * 20 // Higher aggr = looser margin = more pruning
}

/// Quiescence search with delta pruning - only look at captures
fn quiescence(board: &Board, mut alpha: i32, beta: i32, ply: u8) -> i32 {
    // Count qnodes (thread-safe)
    QNODE_COUNT.fetch_add(1, Ordering::Relaxed);
    
    // Depth limit to prevent qsearch explosion in tactical positions
    if ply > 32 {
        return eval::evaluate_lazy(board, alpha, beta);
    }
    
    // Stand pat - use lazy eval for speedup
    let stand_pat = eval::evaluate_lazy(board, alpha, beta);
    
    if stand_pat >= beta {
        return beta;
    }
    if alpha < stand_pat {
        alpha = stand_pat;
    }
    
    // Delta pruning: if we're so far behind that even winning a queen won't help, stop
    const DELTA: i32 = 1000; // Queen value + margin
    if stand_pat + DELTA < alpha {
        return alpha;
    }
    
    // CRITICAL FIX: Use set_iterator_mask to ONLY generate captures
    // Instead of generating all moves then filtering (expensive!)
    let targets = board.color_combined(!board.side_to_move()); // Enemy pieces only
    let mut gen = MoveGen::new_legal(board);
    gen.set_iterator_mask(*targets);  // Only iterate captures!
    let mut captures: Vec<ChessMove> = gen.collect();
    
    // Sort by MVV-LVA
    captures.sort_by_key(|m| -eval::mvv_lva_score(board, *m));
    
    for mv in captures {
        // SEE Pruning: Skip bad captures
        // Unless it's a very valuable capture (Queen), pruning usually safe
        // SEE < 0 means we lose material
        if eval::see(board, mv) < 0 {
            continue;
        }

        // Delta pruning per move: skip captures that can't improve alpha
        let captured_value = match board.piece_on(mv.get_dest()) {
            Some(chess::Piece::Pawn) => 100,
            Some(chess::Piece::Knight) => 320,
            Some(chess::Piece::Bishop) => 330,
            Some(chess::Piece::Rook) => 500,
            Some(chess::Piece::Queen) => 900,
            _ => 0,
        };
        if stand_pat + captured_value + 200 < alpha {
            continue; // This capture can't possibly improve alpha
        }
        
        let new_board = board.make_move_new(mv);
        let score = -quiescence(&new_board, -beta, -alpha, ply + 1);
        
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }
    
    alpha
}

/// Negamax with alpha-beta pruning, NMP, LMR, incremental eval, and continuation history
fn negamax(
    board: &Board,
    eval_state: eval::EvalState,  // Incremental evaluation state
    prev_move: Option<ChessMove>, // Previous move for continuation history
    depth: u8,
    mut alpha: i32,
    beta: i32,
    ply: u8,
    null_ok: bool,  // Can we try null move?
    time_manager: Option<TimeManager>,
) -> (i32, Option<ChessMove>) {
    // Count nodes (thread-safe)
    let nodes = NODE_COUNT.fetch_add(1, Ordering::Relaxed);
    
    // Check time every 2048 nodes
    if nodes & 2047 == 0 {
        if STOP_SEARCH.load(Ordering::Relaxed) {
            return (0, None);
        }
        if let Some(tm) = time_manager {
            if tm.check_time() {
                STOP_SEARCH.store(true, Ordering::Relaxed);
                return (0, None);
            }
        }
    }
    
    if STOP_SEARCH.load(Ordering::Relaxed) {
        return (0, None);
    }
    
    // Debug: Log entry at low ply for diagnostics
    if is_debug() && ply <= 2 {
        eprintln!("[DEBUG] negamax: ply={} depth={} alpha={} beta={}", ply, depth, alpha, beta);
    }
    
    // Base case: quiescence search
    if depth == 0 {
        let q_score = quiescence(board, alpha, beta, 0);
        if is_debug() && ply <= 2 {
            eprintln!("[DEBUG] negamax: depth=0, qsearch returned {}", q_score);
        }
        return (q_score, None);
    }
    
    // NOTE: We removed board.status() here! It was generating all moves
    // internally just to check mate/stalemate - huge waste!
    // We'll detect mate/stalemate after the move loop instead.
    
    let hash = board.get_hash();
    let in_check = *board.checkers() != chess::EMPTY;
    let phase = eval::game_phase(board);
    let is_endgame = phase < 12;  // Less than half material = endgame
    
    // Probe transposition table
    let tt_result = tt_probe(hash);
    let tt_move: Option<ChessMove> = tt_result.as_ref().and_then(|(e, _)| decode_move(e.best_move, board));
    
    if let Some((entry, flag)) = tt_result {
        if is_debug() && ply <= 1 {
            eprintln!("[DEBUG] TT HIT: hash={:x} depth={} score={} flag={:?}", 
                hash, entry.depth, entry.score, flag);
        }
        if entry.depth >= depth {
            let score = entry.score as i32;
            match flag {
                TTFlag::Exact => {
                    if is_debug() && ply <= 1 {
                        eprintln!("[DEBUG] TT CUTOFF: Exact score={}", score);
                    }
                    return (score, tt_move);
                }
                TTFlag::Alpha => {
                    if score <= alpha {
                        if is_debug() && ply <= 1 {
                            eprintln!("[DEBUG] TT CUTOFF: Alpha score={} <= alpha={}", score, alpha);
                        }
                        return (alpha, tt_move);
                    }
                }
                TTFlag::Beta => {
                    if score >= beta {
                        if is_debug() && ply <= 1 {
                            eprintln!("[DEBUG] TT CUTOFF: Beta score={} >= beta={}", score, beta);
                        }
                        return (beta, tt_move);
                    }
                }
                TTFlag::None => {}
            }
        }
    }
    
    // Static eval for pruning decisions - use INCREMENTAL eval for speedup!
    let static_eval = if !in_check { eval::evaluate_with_state(board, &eval_state, alpha, beta) } else { 0 };
    
    // ==========================================
    // REVERSE FUTILITY PRUNING (Static Null Move Pruning)
    // If static eval is way above beta, this position is winning - prune
    // ==========================================
    if !in_check && depth <= 8 {
        let rfp_margin = 100 * depth as i32;  // More aggressive margin
        if static_eval - rfp_margin >= beta {
            return (beta, None);
        }
    }
    
    // ==========================================
    // RAZORING
    // If static eval is far below alpha at shallow depth, drop to qsearch
    // ==========================================
    if !in_check && depth <= 3 && tt_move.is_none() {
        let razor_margin = 300 + 200 * depth as i32;
        if static_eval + razor_margin < alpha {
            let qscore = quiescence(board, alpha, beta, 0);
            if qscore < alpha {
                return (alpha, None);
            }
        }
    }
    
    // ==========================================
    // NULL MOVE PRUNING (NMP)
    // Skip our turn - if opponent still can't beat beta, position is winning
    // Disabled in endgame (zugzwang risk) and when in check
    // ==========================================
    if null_ok && !in_check && !is_endgame && depth >= 3 {
        // Make null move (pass turn)
        if let Some(null_board) = board.null_move() {
            let r = 2 + depth / 4;  // Reduction factor
            // Note: eval_state doesn't change on null move (no piece moved)
            let (score, _) = negamax(&null_board, eval_state, None, depth.saturating_sub(1 + r), -beta, -beta + 1, ply + 1, false, time_manager);
            let score = -score;
            
            if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
            
            if score >= beta {
                return (beta, None);  // Prune!
            }
        }
    }
    
    // ==========================================
    // PROBCUT
    // If a shallow search with a shifted window triggers a cutoff, assume deep search will too
    // ==========================================
    if !in_check && depth >= 5 && tt_move.is_some() {
        let probcut_beta = beta + 200;
        let probcut_depth = depth - 4;
        
        // Shallow search with wider window
        let (score, _) = negamax(board, eval_state, None, probcut_depth, probcut_beta - 1, probcut_beta, ply + 1, false, time_manager);
        let score = -score;
        
        if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
        
        if score >= probcut_beta {
            return (beta, None); // Probable beta cutoff
        }
    }
    
    // ==========================================
    // STAGED MOVE GENERATION
    // Try TT move first - if it cuts off, we save move generation!
    // ==========================================
    
    let killers = KILLERS.with(|k| k.borrow()[(ply as usize).min(63)]);
    let mut best_move: Option<ChessMove> = None;
    let mut best_score = -INFINITY;
    let original_alpha = alpha;
    let mut moves_searched = 0;
    
    // INTERNAL ITERATIVE DEEPENING (IID)
    // If we have no TT move and depth is high, do a shallow search to get a best move hint
    // This improves move ordering significantly at PV nodes
    let mut tt_move = tt_move; // Make mutable for IID update
    if tt_move.is_none() && depth >= 6 {
         let iid_depth = depth - 2;
         // We ignore result, just want to populate TT
         let _ = negamax(board, eval_state, prev_move, iid_depth, alpha, beta, ply, null_ok, time_manager);
         
         // Re-probe TT to get the move we just found
         if let Some((entry, _)) = tt_probe(hash) {
             tt_move = decode_move(entry.best_move, board);
         }
    }

    // STAGE 1: Try TT move first (if valid and legal)
    if let Some(tt_mv) = tt_move {
        // Verify TT move is legal by checking if it's in legal moves
        let is_legal = MoveGen::new_legal(board).any(|m| m == tt_mv);
        
        if is_legal {
            // ==========================================
            // SINGULAR EXTENSION CHECK
            // If TT move is clearly better than alternatives, extend search
            // ==========================================
            let mut extension: u8 = 0;
            
            // Only check for singular extension if:
            // - Not in check (extension already happens for check evasions)
            // - TT entry is a lower bound (Beta flag means tt_move >= beta) 
            // - Depth is sufficient (worth the overhead)
            // - We have TT result with good depth
            if !in_check && depth >= 8 {
                if let Some((ref entry, TTFlag::Beta)) = tt_result {
                    if entry.depth >= depth.saturating_sub(3) {
                        let se_beta = (entry.score as i32).saturating_sub(2 * depth as i32);
                        
                        // Search all OTHER moves at reduced depth
                        // If they all fail low, TT move is singular
                        let mut is_singular = true;
                        // Optimization: Don't generate all moves, just iterate
                        let mut gen = MoveGen::new_legal(board);
                        for other_mv in gen {
                            if other_mv == tt_mv { continue; }
                            
                            let new_board = board.make_move_new(other_mv);
                            let mut new_eval = eval_state;
                            new_eval.apply_move(board, other_mv);
                            let se_depth = depth / 2;
                            // Null window search for singular extension check
                            let (score, _) = negamax(&new_board, new_eval, Some(other_mv), se_depth, -se_beta, -se_beta + 1, ply + 1, false, time_manager);
                            let score = -score;
                            
                            if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
                            
                            if score >= se_beta {
                                // Found a move that's also good - not singular
                                is_singular = false;
                                break;
                            }
                        }
                        
                        if is_singular {
                            extension = 1; // Extend by 1 ply!
                        }
                    }
                }
            }
            
            let new_board = board.make_move_new(tt_mv);
            let mut new_eval = eval_state;
            new_eval.apply_move(board, tt_mv);
            
            // Full window search for first move (with possible extension)
            let search_depth = (depth - 1).saturating_add(extension);
            let (score, _) = negamax(&new_board, new_eval, Some(tt_mv), search_depth, -beta, -alpha, ply + 1, true, time_manager);
            let score = -score;
            
            if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
            
            if score > best_score {
                best_score = score;
                best_move = Some(tt_mv);
            }
            
            if score > alpha {
                alpha = score;
            }
            
            // Beta cutoff! Saved ALL move generation!
            if score >= beta {
                // Update killers and history for quiet moves
                if !eval::is_capture(board, tt_mv) {
                    update_killers(ply, tt_mv);
                    update_history(tt_mv, depth);
                    update_counter_history(prev_move, tt_mv, depth); // Continuation history!
                }
                tt_store(hash, depth, score, TTFlag::Beta, Some(tt_mv));
                return (beta, Some(tt_mv));
            }
            
            moves_searched = 1;
        }
    }
    
    // STAGE 2: Generate and search remaining moves
    let mut moves: Vec<ChessMove> = Vec::with_capacity(64);
    moves.extend(MoveGen::new_legal(board).filter(|m| Some(*m) != tt_move));
    
    if moves.is_empty() && moves_searched == 0 {
        // No legal moves at all - either checkmate or stalemate
        if in_check {
            return (-MATE_SCORE + ply as i32, None);  // Checkmate
        } else {
            // DYNAMIC CONTEMPT for stalemate:
            // - If we're winning (static_eval > 0): draw is BAD (return negative)
            // - If we're losing (static_eval < 0): draw is GOOD (return positive)
            // - If equal: draw is neutral (return 0)
            let dynamic_contempt = if static_eval > 100 {
                -20  // We're winning, avoid draw!
            } else if static_eval < -100 {
                20   // We're losing, draw is escape!
            } else {
                0    // Equal position, draw is fine
            };
            return (dynamic_contempt, None);
        }
    }
    
    // Score and sort remaining moves
    let mut scored_moves: Vec<(ChessMove, i32)> = moves.iter().map(|m| {
        let score = if Some(*m) == killers[0] || Some(*m) == killers[1] {
            500_000 // Killer moves
        } else if eval::is_capture(board, *m) {
            // Use SEE to distinguish good/bad captures
            let see = eval::see(board, *m);
            if see >= 0 {
                // Good capture: High priority (above killers)
                // Base 2,000,000 + MVV-LVA + SEE
                2_000_000 + eval::mvv_lva_score(board, *m) + see
            } else {
                // Bad capture: Low priority (below killers, but above bad quiets)
                -100_000 + see // Negative score
            }
        } else {
            // History heuristic + continuation history for quiet moves
            let from = m.get_source().to_index();
            let to = m.get_dest().to_index();
            let history_score = HISTORY.with(|h| h.borrow()[from][to] / 100);
            let counter_score = get_counter_history(prev_move, *m) / 100;
            history_score + counter_score
        };
        (*m, score)
    }).collect();
    
    scored_moves.sort_by_key(|(_, score)| -score);
    let moves: Vec<ChessMove> = scored_moves.into_iter().map(|(m, _)| m).collect();
    
    // Set best_move to first if not yet set
    if best_move.is_none() && !moves.is_empty() {
        best_move = Some(moves[0]);
    }
    
    // LMP thresholds by depth
    let lmp_threshold = [0, 5, 8, 12, 16, 20, 24, 28][depth.min(7) as usize];  // More moves searched
    
    for mv in moves.iter() {
        let mv = *mv;
        let i = moves_searched; // Use total moves searched for LMP
        moves_searched += 1;
        let is_capture = eval::is_capture(board, mv);
        
        // ==========================================
        // LATE MOVE PRUNING (LMP)
        // Skip late quiet moves entirely at shallow depths
        // ==========================================
        if depth <= 5 && !in_check && !is_capture && i >= lmp_threshold && i > 0 {
            continue;  // Only prune at shallow depths
        }
        
        // ==========================================
        // FUTILITY PRUNING - expanded to depth 6
        // Skip moves that can't improve alpha at shallow depths
        // ==========================================
        if depth <= 4 && !in_check && !is_capture && i > 0 {
            let margin = 200 * depth as i32;  // Wider margin = safer
            if static_eval + margin < alpha {
                continue; // Prune this move
            }
        }
        
        let new_board = board.make_move_new(mv);
        let mut new_eval = eval_state;
        new_eval.apply_move(board, mv);
        let gives_check = *new_board.checkers() != chess::EMPTY;
        
        // ==========================================
        // LATE MOVE REDUCTIONS (LMR)
        // ==========================================
        let mut reduction = 0u8;
        if i >= 2 && depth >= 2 && !is_capture && !gives_check && !in_check {
            // Logarithmic reduction formula: R = 1 + ln(depth) * ln(i) / 2
            // This is standard in Ethereal, Stockfish, etc.
            let base_reduction = 1.0 + (depth as f32).ln() * ((i) as f32).ln() / 2.0;
            reduction = base_reduction as u8;
            
            // Adjust based on history: if move is historically bad, reduce MORE
            // History ranges 0..10000. 
            // Good move (>5000) -> reduce less. Bad move (<2000) -> reduce more.
            let from = mv.get_source().to_index();
            let to = mv.get_dest().to_index();
            let hist_score = HISTORY.with(|h| h.borrow()[from][to]);
            
            if hist_score > 8000 {
                reduction = reduction.saturating_sub(1);
            } else if hist_score < 1000 {
                reduction = reduction.saturating_add(1);
            }
            
            // Safety
            reduction = reduction.min(depth - 1).max(0);
        }
        
        // ==========================================
        // PRINCIPAL VARIATION SEARCH (PVS)
        // Search first move with full window, others with null window
        // ==========================================
        let new_depth = (depth as i16 - 1 - reduction as i16).max(0) as u8;
        let mut score;
        
        if i == 0 {
            // First move: full window search
            let (s, _) = negamax(&new_board, new_eval, Some(mv), new_depth, -beta, -alpha, ply + 1, true, time_manager);
            score = -s;
        } else {
            // Null window search (scout)
            let (s, _) = negamax(&new_board, new_eval, Some(mv), new_depth, -alpha - 1, -alpha, ply + 1, true, time_manager);
            score = -s;
            
            if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
            
            // Re-search with full window if it improved alpha
            if score > alpha && score < beta {
                let (s, _) = negamax(&new_board, new_eval, Some(mv), new_depth, -beta, -alpha, ply + 1, true, time_manager);
                score = -s;
            }
        }
        
        // Re-search if LMR reduced search looks good
        if reduction > 0 && score > alpha {
            // Full depth search
            let (rescore, _) = negamax(&new_board, new_eval, Some(mv), depth - 1, -beta, -alpha, ply + 1, true, time_manager);
            score = -rescore;
        }
        
        if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
        
        if score >= beta {
            // Store killer move and update history (only quiet moves)
            if !is_capture {
                let ply_idx = (ply as usize).min(63);  // Clamp to valid range
                KILLERS.with(|k| {
                    let mut killers = k.borrow_mut();
                    if killers[ply_idx][0] != Some(mv) {
                        killers[ply_idx][1] = killers[ply_idx][0];
                        killers[ply_idx][0] = Some(mv);
                    }
                });
                // Update history heuristic
                let from = mv.get_source().to_index();
                let to = mv.get_dest().to_index();
                HISTORY.with(|h| {
                    let mut hist = h.borrow_mut();
                    hist[from][to] += (depth as i32) * (depth as i32);
                    // Cap history to prevent overflow
                    if hist[from][to] > 10000 {
                        hist[from][to] = 10000;
                    }
                });
                // Update continuation history
                update_counter_history(prev_move, mv, depth);
            }
            
            // Store in TT
            tt_store(hash, depth, beta, TTFlag::Beta, Some(mv));
            
            return (beta, Some(mv));
        }
        
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
        
        if score > alpha {
            alpha = score;
        }
    }
    
    // Store in TT
    let flag = if alpha > original_alpha { TTFlag::Exact } else { TTFlag::Alpha };
    tt_store(hash, depth, best_score, flag, best_move);
    
    (best_score, best_move)
}

/// Iterative deepening search with aspiration windows
/// aggressiveness: 1-10, higher = more pruning = faster but riskier
pub fn iterative_deepening(
    board: &Board, 
    max_depth: u8, 
    aggressiveness: u8,
    wtime: Option<u64>,
    btime: Option<u64>,
    movestogo: Option<u64>
) -> (Option<ChessMove>, i32) {
    // Set aggressiveness for this search
    AGGRESSIVENESS.with(|a| a.set(aggressiveness));
    
    reset_node_counts();  // Reset node counters for this search
    KILLERS.with(|k| *k.borrow_mut() = [[None; 2]; 64]);
    // Age history table instead of clearing (keeps some knowledge)
    HISTORY.with(|h| {
        let mut hist = h.borrow_mut();
        for i in 0..64 {
            for j in 0..64 {
                hist[i][j] /= 2; // Decay by 50%
            }
        }
    });
    
    // Initialize Time Manager
    let time_manager = TimeManager::new(wtime, btime, movestogo, board.side_to_move());
    
    if is_debug() {
        eprintln!("[DEBUG] iterative_deepening: starting max_depth={} aggr={}", max_depth, aggressiveness);
    }
    
    let mut best_move = None;
    let mut best_score = 0;
    
    // Initialize incremental eval state once for root position
    let root_eval = eval::EvalState::new(board);
    
    for depth in 1..=max_depth {
        if is_debug() {
            eprintln!("[DEBUG] ID: starting depth {}/{}", depth, max_depth);
        }
        
        // Check time before starting new depth
        if let Some(tm) = time_manager {
            // If we already used > 50% of time, don't start new depth
            // unless depth is very low
            if depth > 4 && tm.start_time.elapsed().as_millis() > tm.allocated_time / 2 {
                break;
            }
        }
        
        // Aspiration window search
        let window = 50; // Start with narrow window
        let alpha = if depth > 1 { best_score - window } else { -INFINITY };
        let beta = if depth > 1 { best_score + window } else { INFINITY };
        
        let (mut score, mut mv) = negamax(board, root_eval, None, depth, alpha, beta, 0, true, time_manager);
        
        // If search was stopped by time, break and return previous best
        if STOP_SEARCH.load(Ordering::Relaxed) {
            break;
        }
        
        // Re-search with wider window if score is outside bounds
        if score <= alpha || score >= beta {
            // Score outside aspiration window, re-search with full window
            let result = negamax(board, root_eval, None, depth, -INFINITY, INFINITY, 0, true, time_manager);
            score = result.0;
            mv = result.1;
            
            if STOP_SEARCH.load(Ordering::Relaxed) {
                break;
            }
        }
        
        if let Some(m) = mv {
            best_move = Some(m);
            best_score = score;
        }
        
        // Print NPS info for last depth (diagnostics)
        let (nodes, qnodes) = get_node_counts();
        eprintln!("info depth {} nodes {} qnodes {} total {}", 
            depth, nodes, qnodes, nodes + qnodes);
    }
    
    (best_move, best_score)
}

/// Parallel search at root level using rayon with ITERATIVE DEEPENING
/// Without ID, jumping straight to depth 12+ with empty TT = brute force = death
pub fn parallel_root_search(
    board: &Board, 
    max_depth: u8, 
    aggressiveness: u8,
    wtime: Option<u64>,
    btime: Option<u64>,
    movestogo: Option<u64>
) -> (Option<ChessMove>, i32) {
    AGGRESSIVENESS.with(|a| a.set(aggressiveness));
    KILLERS.with(|k| *k.borrow_mut() = [[None; 2]; 64]);
    reset_node_counts();
    
    let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
    if moves.is_empty() {
        return (None, 0);
    }
    
    let mut best_move = None;
    let mut best_score = 0;
    
    let time_manager = TimeManager::new(wtime, btime, movestogo, board.side_to_move());
    
    // Initialize incremental eval state for root position
    let root_eval = eval::EvalState::new(board);
    
    // ITERATIVE DEEPENING LOOP (Critical for Move Ordering!)
    for depth in 1..=max_depth {
        // Check time
        if let Some(tm) = time_manager {
            if tm.check_time() { break; }
        }
        
        let mut current_best_move = moves[0];
        let mut current_best_score = -INFINITY;
        
        // Search each root move in parallel at this depth
        let results: Vec<(ChessMove, i32)> = moves
            .par_iter()
            .map(|&mv| {
                if STOP_SEARCH.load(Ordering::Relaxed) {
                    return (mv, -INFINITY);
                }
                
                // Thread-local time check
                if let Some(tm) = time_manager {
                    if tm.check_time() {
                        STOP_SEARCH.store(true, Ordering::Relaxed);
                        return (mv, -INFINITY);
                    }
                }
                
                let new_board = board.make_move_new(mv);
                let mut new_eval = root_eval;
                new_eval.apply_move(board, mv);
                let (score, _) = negamax(&new_board, new_eval, Some(mv), depth - 1, -INFINITY, INFINITY, 1, true, time_manager);
                (mv, -score)
            })
            .collect();
            
        if STOP_SEARCH.load(Ordering::Relaxed) {
            break;
        }
        
        // Find best move at this depth
        for (mv, score) in results {
            if score > current_best_score {
                current_best_score = score;
                current_best_move = mv;
            }
        }
        
        best_move = Some(current_best_move);
        best_score = current_best_score;
        
        // Store best move in TT for next iteration's move ordering!
        let hash = board.get_hash();
        tt_store(hash, depth, best_score, TTFlag::Exact, best_move);
        
        // Print info for diagnostics
        eprintln!("info depth {} score cp {} pv {}", depth, best_score, current_best_move);
    }
    
    (best_move, best_score)
}


// NOTE: simple_negamax removed - dead code, parallel_root_search now uses real negamax with ID
