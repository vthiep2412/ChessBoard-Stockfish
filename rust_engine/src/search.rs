//! Alpha-Beta Search with Transposition Table, Iterative Deepening, and Parallel Search

use chess::{Board, MoveGen, ChessMove, BoardStatus};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use crate::eval;
use crate::movegen::{StagedMoveGen, Stage};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

// =============================================================================
// TRANSPOSITION TABLE - Fixed-size array for O(1) access
// =============================================================================

/// TT size: 2^22 = 4,194,304 entries (~64MB with 16 bytes per entry)
const TT_SIZE: usize = 1 << 22;
const TT_MASK: usize = TT_SIZE - 1;

/// Compact TT entry - 16 bytes total
#[derive(Clone, Copy)]
#[repr(C)]
pub struct TTEntry {
    hash: u64,              // Full hash for collision detection
    pub best_move: u16,         // Encoded move (from 6 bits + to 6 bits + promo 4 bits)
    pub score: i16,             // Score (clamped to i16 range)
    pub depth: u8,              // Search depth
    pub flag: u8,               // 0=None, 1=Exact, 2=Alpha, 3=Beta
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
pub enum TTFlag {
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
pub fn encode_move(mv: ChessMove) -> u16 {
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
pub fn decode_move(encoded: u16, _board: &Board) -> Option<ChessMove> {
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
pub fn tt_probe(hash: u64) -> Option<(TTEntry, TTFlag)> {
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
thread_local! {
    pub static KILLERS: std::cell::RefCell<[[Option<ChessMove>; 2]; 64]> =
        std::cell::RefCell::new([[None; 2]; 64]);
}

/// History heuristic table [from_sq][to_sq] for move ordering
thread_local! {
    pub static HISTORY: std::cell::RefCell<[[i32; 64]; 64]> =
        std::cell::RefCell::new([[0; 64]; 64]);
}

/// Counter-Move History: indexed by PREVIOUS move's [from][to]
/// Optimized to use Heap allocation (Box) to avoid stack overflow with thread_local
thread_local! {
    pub static COUNTER_HISTORY: std::cell::RefCell<Box<[[[[i16; 64]; 64]; 64]; 64]>> =
        std::cell::RefCell::new(Box::new([[[[0; 64]; 64]; 64]; 64]));
}

/// Aggressiveness level (1-10) for pruning - set by iterative_deepening
thread_local! {
    static AGGRESSIVENESS: std::cell::Cell<u8> = std::cell::Cell::new(5);
}

const INFINITY: i32 = 30000;
const MATE_SCORE: i32 = 29000;
const CONTEMPT: i32 = -15; // Slight draw aversion (engine sees draws as bad)

static NODE_COUNT: AtomicU64 = AtomicU64::new(0);
static QNODE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static STOP_SEARCH: AtomicBool = AtomicBool::new(false);

/// Helper to stop search
pub fn stop() {
    STOP_SEARCH.store(true, Ordering::Relaxed);
}

/// Helper to clear stop flag (explicitly requested by review)
pub fn clear_stop_flag() {
    STOP_SEARCH.store(false, Ordering::Relaxed);
}

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
    
    // Delta pruning
    const DELTA: i32 = 1000;
    if stand_pat + DELTA < alpha {
        return alpha;
    }
    
    let mut targets = *board.color_combined(!board.side_to_move());
    if let Some(ep_sq) = board.en_passant() {
        // En-passant square IS the destination square in chess crate
        targets |= chess::BitBoard::from_square(ep_sq);
    }

    let mut gen = MoveGen::new_legal(board);
    gen.set_iterator_mask(targets);
    let mut captures: Vec<ChessMove> = gen.collect();
    
    // Sort by MVV-LVA
    captures.sort_by_key(|m| -eval::mvv_lva_score(board, *m));
    
    for mv in captures {
        if eval::see(board, mv) < 0 {
            continue;
        }

        let captured_value = match board.piece_on(mv.get_dest()) {
            Some(chess::Piece::Pawn) => 100,
            Some(chess::Piece::Knight) => 320,
            Some(chess::Piece::Bishop) => 330,
            Some(chess::Piece::Rook) => 500,
            Some(chess::Piece::Queen) => 900,
            _ => 0,
        };
        if stand_pat + captured_value + 200 < alpha {
            continue;
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
    eval_state: eval::EvalState,
    prev_move: Option<ChessMove>,
    depth: u8,
    mut alpha: i32,
    beta: i32,
    ply: u8,
    null_ok: bool,
    time_manager: Option<TimeManager>,
) -> (i32, Option<ChessMove>) {
    let nodes = NODE_COUNT.fetch_add(1, Ordering::Relaxed);
    
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
    
    if depth == 0 {
        let q_score = quiescence(board, alpha, beta, 0);
        return (q_score, None);
    }
    
    let hash = board.get_hash();
    let in_check = *board.checkers() != chess::EMPTY;
    let phase = eval::game_phase(board);
    let is_endgame = phase < 12;
    
    let tt_result = tt_probe(hash);
    let tt_move: Option<ChessMove> = tt_result.as_ref().and_then(|(e, _)| decode_move(e.best_move, board));
    
    if let Some((entry, flag)) = tt_result {
        if entry.depth >= depth {
            let score = entry.score as i32;
            match flag {
                TTFlag::Exact => return (score, tt_move),
                TTFlag::Alpha => if score <= alpha { return (alpha, tt_move); },
                TTFlag::Beta => if score >= beta { return (beta, tt_move); },
                TTFlag::None => {}
            }
        }
    }
    
    let static_eval = if !in_check { eval::evaluate_with_state(board, &eval_state, alpha, beta) } else { 0 };
    
    // Reverse Futility Pruning
    if !in_check && depth <= 8 {
        let rfp_margin = 100 * depth as i32;
        if static_eval - rfp_margin >= beta {
            return (beta, None);
        }
    }
    
    // Razoring
    if !in_check && depth <= 3 && tt_move.is_none() {
        let razor_margin = 300 + 200 * depth as i32;
        if static_eval + razor_margin < alpha {
            let qscore = quiescence(board, alpha, beta, 0);
            if qscore < alpha {
                return (alpha, None);
            }
        }
    }
    
    // Null Move Pruning
    if null_ok && !in_check && !is_endgame && depth >= 3 {
        if let Some(null_board) = board.null_move() {
            let r = 2 + depth / 4;
            let (score, _) = negamax(&null_board, eval_state, None, depth.saturating_sub(1 + r), -beta, -beta + 1, ply + 1, false, time_manager);
            let score = -score;
            if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
            if score >= beta {
                return (beta, None);
            }
        }
    }
    
    // ProbCut
    if !in_check && depth >= 5 && tt_move.is_some() {
        let probcut_beta = beta + 200;
        let probcut_depth = depth - 4;

        // Search captures with reduced depth/window
        let mut gen = MoveGen::new_legal(board);
        let targets = board.color_combined(!board.side_to_move());
        gen.set_iterator_mask(*targets);

        for mv in gen {
            // Only search if SEE >= 0 (simple check to avoid bad captures)
            if eval::see(board, mv) < 0 { continue; }

            let new_board = board.make_move_new(mv);
            let mut new_eval = eval_state;
            new_eval.apply_move(&new_board, mv);

            let (score, _) = negamax(&new_board, new_eval, Some(mv), probcut_depth, -probcut_beta, -probcut_beta + 1, ply + 1, false, time_manager);
            let score = -score;

            if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }

            if score >= probcut_beta {
                return (beta, Some(mv));
            }
        }
    }
    
    let mut best_move: Option<ChessMove> = None;
    let mut best_score = -INFINITY;
    let original_alpha = alpha;
    let mut moves_searched = 0;
    
    // Internal Iterative Deepening
    let mut tt_move = tt_move;
    if tt_move.is_none() && depth >= 6 {
         let iid_depth = depth - 2;
         let _ = negamax(board, eval_state, prev_move, iid_depth, alpha, beta, ply, null_ok, time_manager);
         if let Some((entry, _)) = tt_probe(hash) {
             tt_move = decode_move(entry.best_move, board);
         }
    }

    // Singular Extension Check
    let mut extension: u8 = 0;
    if !in_check && depth >= 8 {
        if let Some((ref entry, TTFlag::Beta)) = tt_result {
            if let Some(tm) = tt_move {
                if entry.depth >= depth.saturating_sub(3) {
                    let se_beta = (entry.score as i32).saturating_sub(2 * depth as i32);
                    let se_depth = (depth - 1) / 2;

                    // Search other moves at reduced depth to verify singularity
                    // Optimization: Only check first few moves and use StagedMoveGen
                    let mut is_singular = true;
                    let mut moves_checked = 0;
                    let max_checks = 6; // Limit checks to avoid overhead

                    // We don't have a staged gen for *other* moves easily without excluding TT move inside
                    // But we can use StagedMoveGen and skip the TT move.
                    let mut se_gen = StagedMoveGen::new(board, None, ply);

                    while let Some(other_mv) = se_gen.next() {
                        if other_mv == tm { continue; }

                        moves_checked += 1;
                        if moves_checked > max_checks {
                            is_singular = false; // Assume not singular if we have many alternatives
                            break;
                        }

                        let new_board = board.make_move_new(other_mv);
                        let mut new_eval = eval_state;
                        new_eval.apply_move(&new_board, other_mv);

                        let (score, _) = negamax(&new_board, new_eval, Some(other_mv), se_depth, -se_beta, -se_beta + 1, ply + 1, false, time_manager);
                        let score = -score;

                        if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }

                        if score >= se_beta {
                            is_singular = false;
                            break;
                        }
                    }

                    if is_singular {
                        extension = 1;
                    }
                }
            }
        }
    }

    let mut move_gen = StagedMoveGen::new(board, tt_move, ply);
    
    while let Some(mv) = move_gen.next() {
        let i = moves_searched;
        moves_searched += 1;
        let is_capture = eval::is_capture(board, mv);
        
        let lmp_threshold = [0, 5, 8, 12, 16, 20, 24, 28][depth.min(7) as usize];
        if depth <= 5 && !in_check && !is_capture && i >= lmp_threshold && i > 0 {
            continue;
        }
        
        if depth <= 4 && !in_check && !is_capture && i > 0 {
            let margin = 200 * depth as i32;
            if static_eval + margin < alpha {
                continue;
            }
        }
        
        let new_board = board.make_move_new(mv);
        let mut new_eval = eval_state;
        new_eval.apply_move(&new_board, mv);
        let gives_check = *new_board.checkers() != chess::EMPTY;
        
        let mut reduction = 0u8;
        if i >= 2 && depth >= 2 && !is_capture && !gives_check && !in_check {
            let base_reduction = 1.0 + (depth as f32).ln() * ((i) as f32).ln() / 2.0;
            reduction = base_reduction as u8;
            
            let from = mv.get_source().to_index();
            let to = mv.get_dest().to_index();
            let hist_score = HISTORY.with(|h| h.borrow()[from][to]);
            
            if hist_score > 8000 {
                reduction = reduction.saturating_sub(1);
            } else if hist_score < 1000 {
                reduction = reduction.saturating_add(1);
            }
            
            reduction = reduction.min(depth - 1).max(0);
        }
        
        // Apply extension to new depth
        let extend = if i == 0 { extension } else { 0 };
        let new_depth = (depth as i16 - 1 - reduction as i16 + extend as i16).max(0) as u8;
        let mut score;
        
        if i == 0 {
             let (s, _) = negamax(&new_board, new_eval, Some(mv), new_depth, -beta, -alpha, ply + 1, true, time_manager);
             score = -s;
        } else {
             let (s, _) = negamax(&new_board, new_eval, Some(mv), new_depth, -alpha - 1, -alpha, ply + 1, true, time_manager);
             score = -s;

             if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }

             if score > alpha && score < beta {
                 let (s, _) = negamax(&new_board, new_eval, Some(mv), new_depth, -beta, -alpha, ply + 1, true, time_manager);
                 score = -s;
             }
        }
        
        if reduction > 0 && score > alpha {
             let (rescore, _) = negamax(&new_board, new_eval, Some(mv), depth - 1, -beta, -alpha, ply + 1, true, time_manager);
             score = -rescore;
        }
        
        if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
        
        if score >= beta {
             if !is_capture {
                 update_killers(ply, mv);
                 update_history(mv, depth);
                 update_counter_history(prev_move, mv, depth);
             }
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
    
    if moves_searched == 0 {
        if in_check {
            return (-MATE_SCORE + ply as i32, None);
        } else {
            let dynamic_contempt = if static_eval > 100 { -20 } else if static_eval < -100 { 20 } else { 0 };
            return (dynamic_contempt, None);
        }
    }

    let flag = if alpha > original_alpha { TTFlag::Exact } else { TTFlag::Alpha };
    tt_store(hash, depth, best_score, flag, best_move);
    
    (best_score, best_move)
}

/// Iterative deepening search with aspiration windows
pub fn iterative_deepening(
    board: &Board, 
    max_depth: u8, 
    aggressiveness: u8,
    wtime: Option<u64>,
    btime: Option<u64>,
    movestogo: Option<u64>
) -> (Option<ChessMove>, i32) {
    AGGRESSIVENESS.with(|a| a.set(aggressiveness));
    
    reset_node_counts();
    KILLERS.with(|k| *k.borrow_mut() = [[None; 2]; 64]);
    HISTORY.with(|h| {
        let mut hist = h.borrow_mut();
        for i in 0..64 {
            for j in 0..64 {
                hist[i][j] /= 2;
            }
        }
    });
    
    let time_manager = TimeManager::new(wtime, btime, movestogo, board.side_to_move());
    
    if is_debug() {
        eprintln!("[DEBUG] iterative_deepening: starting max_depth={} aggr={}", max_depth, aggressiveness);
    }
    
    let mut best_move = None;
    let mut best_score = 0;
    
    let root_eval = eval::EvalState::new(board);
    
    for depth in 1..=max_depth {
        if is_debug() {
            eprintln!("[DEBUG] ID: starting depth {}/{}", depth, max_depth);
        }
        
        if let Some(tm) = time_manager {
            if depth > 4 && tm.start_time.elapsed().as_millis() > tm.allocated_time / 2 {
                break;
            }
        }
        
        let window = 50;
        let alpha = if depth > 1 { best_score - window } else { -INFINITY };
        let beta = if depth > 1 { best_score + window } else { INFINITY };
        
        let (mut score, mut mv) = negamax(board, root_eval, None, depth, alpha, beta, 0, true, time_manager);
        
        if STOP_SEARCH.load(Ordering::Relaxed) {
            break;
        }
        
        if score <= alpha || score >= beta {
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
        
        let (nodes, qnodes) = get_node_counts();
        println!("info depth {} score cp {} nodes {} qnodes {} total {}",
            depth, best_score, nodes, qnodes, nodes + qnodes);
    }
    
    (best_move, best_score)
}

/// Parallel search at root level using rayon with ITERATIVE DEEPENING
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
    
    let root_eval = eval::EvalState::new(board);
    
    for depth in 1..=max_depth {
        if let Some(tm) = time_manager {
            if tm.check_time() { break; }
        }
        
        let mut current_best_move = moves[0];
        let mut current_best_score = -INFINITY;
        
        let results: Vec<(ChessMove, i32)> = moves
            .par_iter()
            .map(|&mv| {
                if STOP_SEARCH.load(Ordering::Relaxed) {
                    return (mv, -INFINITY);
                }
                
                if let Some(tm) = time_manager {
                    if tm.check_time() {
                        STOP_SEARCH.store(true, Ordering::Relaxed);
                        return (mv, -INFINITY);
                    }
                }
                
                let new_board = board.make_move_new(mv);
                let mut new_eval = root_eval;
                new_eval.apply_move(&new_board, mv);
                let (score, _) = negamax(&new_board, new_eval, Some(mv), depth - 1, -INFINITY, INFINITY, 1, true, time_manager);
                (mv, -score)
            })
            .collect();
            
        if STOP_SEARCH.load(Ordering::Relaxed) {
            break;
        }
        
        for (mv, score) in results {
            if score > current_best_score {
                current_best_score = score;
                current_best_move = mv;
            }
        }
        
        best_move = Some(current_best_move);
        best_score = current_best_score;
        
        let hash = board.get_hash();
        tt_store(hash, depth, best_score, TTFlag::Exact, best_move);
        
        println!("info depth {} score cp {} pv {}", depth, best_score, current_best_move);
    }
    
    (best_move, best_score)
}
