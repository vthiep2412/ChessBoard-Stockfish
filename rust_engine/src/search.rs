//! Alpha-Beta Search with Transposition Table, Iterative Deepening, and Parallel Search

use chess::{Board, MoveGen, ChessMove};
use std::thread;
use crate::eval;
use crate::movegen::{StagedMoveGen};
use crate::tablebase;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

// =============================================================================
// TRANSPOSITION TABLE - Fixed-size array for O(1) access
// =============================================================================

/// TT size: 2^22 = 4,194,304 entries (~64MB with 16 bytes per entry)
const TT_SIZE: usize = 1 << 22;
const TT_MASK: usize = TT_SIZE - 1;
const MAX_PLY: u8 = 100; // Limit recursion to prevent Stack Overflow

/// Atomic TT entry - 16 bytes total (2 x u64)
#[repr(C)]
pub struct TTEntry {
    hash: AtomicU64,
    data: AtomicU64,
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            hash: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

/// Unpacked TT Data
#[derive(Clone, Copy)]
pub struct TTEntryData {
    pub hash: u64,
    pub best_move: u16,
    pub score: i16,
    pub depth: u8,
    pub flag: u8,
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
#[inline(always)]
pub fn decode_move(encoded: u16) -> Option<ChessMove> {
    if encoded == 0 {
        return None;
    }
    
    let from_idx = (encoded & 0x3F) as u8;
    let to_idx = ((encoded >> 6) & 0x3F) as u8;
    
    if from_idx >= 64 || to_idx >= 64 {
        return None;
    }
    
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
// =============================================================================

use std::sync::OnceLock;

// Safe global TT using OnceLock
static GLOBAL_TT: OnceLock<Vec<TTEntry>> = OnceLock::new();

#[inline(always)]
fn get_tt() -> &'static [TTEntry] {
    GLOBAL_TT.get_or_init(|| {
        let mut v = Vec::with_capacity(TT_SIZE);
        v.resize_with(TT_SIZE, TTEntry::default);
        v
    })
}

/// Probe the transposition table - LOCKLESS & SAFE
#[inline(always)]
pub fn tt_probe(hash: u64) -> Option<(TTEntryData, TTFlag)> {
    let idx = (hash as usize) & TT_MASK;
    // Safe because TT_MASK ensures bounds and OnceLock ensures initialization
    let entry = unsafe { get_tt().get_unchecked(idx) };

    // Read Key -> Data -> Key
    let key = entry.hash.load(Ordering::Acquire);
    if key != hash {
        return None;
    }

    let data = entry.data.load(Ordering::Acquire);

    let key2 = entry.hash.load(Ordering::Acquire);
    if key2 != key {
        return None;
    }

    let best_move = (data & 0xFFFF) as u16;
    let score = ((data >> 16) & 0xFFFF) as i16;
    let depth_entry = ((data >> 32) & 0xFF) as u8;
    let flag_val = ((data >> 40) & 0xFF) as u8;

    if flag_val != 0 {
        let unpacked = TTEntryData {
            hash: key,
            best_move,
            score,
            depth: depth_entry,
            flag: flag_val,
        };
        Some((unpacked, TTFlag::from(flag_val)))
    } else {
        None
    }
}

/// Store an entry in the transposition table - LOCKLESS & SAFE
#[inline(always)]
fn tt_store(hash: u64, depth: u8, score: i32, flag: TTFlag, best_move: Option<ChessMove>) {
    let idx = (hash as usize) & TT_MASK;
    let entry = unsafe { get_tt().get_unchecked(idx) };

    let old_data = entry.data.load(Ordering::Relaxed);
    let old_depth = ((old_data >> 32) & 0xFF) as u8;
    let old_hash = entry.hash.load(Ordering::Relaxed);

    // Replacement strategy: Always replace if new search is deeper,
    // or if the slot is empty.
    // We do NOT blindly replace same-hash unless it is deeper.
    // This prevents deeper entries from being overwritten by shallower searches (CodeRabbit fix).
    if depth >= old_depth || old_hash == 0 {
        let mv_encoded = best_move.map(encode_move).unwrap_or(0) as u64;
        let score_encoded = (score.clamp(-30000, 30000) as i16 as u16) as u64;
        let depth_encoded = depth as u64;
        let flag_encoded = flag as u64;
        
        let new_data = mv_encoded
                     | (score_encoded << 16)
                     | (depth_encoded << 32)
                     | (flag_encoded << 40);

        // Safe Write Pattern: Invalidate Key -> Write Data -> Write Key
        entry.hash.store(0, Ordering::Release);
        entry.data.store(new_data, Ordering::Release);
        entry.hash.store(hash, Ordering::Release);
    }
}

pub static DEBUG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_debug(enabled: bool) {
    DEBUG.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn is_debug() -> bool {
    DEBUG.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn tt_clear() {
    // Only clear if initialized
    if let Some(entries) = GLOBAL_TT.get() {
        for entry in entries {
            entry.hash.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
        // Increment Epoch to signal thread-local history clearing
        HISTORY_EPOCH.fetch_add(1, Ordering::Relaxed);

        if is_debug() {
            eprintln!("[DEBUG] TT cleared - {} entries zeroed", TT_SIZE);
        }
    }
}

thread_local! {
    pub static KILLERS: std::cell::RefCell<[[Option<ChessMove>; 2]; 64]> =
        std::cell::RefCell::new([[None; 2]; 64]);
}

thread_local! {
    pub static HISTORY: std::cell::RefCell<[[i32; 64]; 64]> =
        std::cell::RefCell::new([[0; 64]; 64]);
}

thread_local! {
    pub static COUNTER_HISTORY: std::cell::RefCell<Box<[[[[i16; 64]; 64]; 64]; 64]>> =
        std::cell::RefCell::new({
            // Use Vec to allocate on heap, then convert to Box<[T; N]> to avoid stack overflow
            // Element size is 64*64*64*2 = 512KB, which fits on stack. Total 32MB fits on heap.
            let v = vec![[[[0; 64]; 64]; 64]; 64];
            v.into_boxed_slice().try_into().unwrap()
        });
}

thread_local! {
    static AGGRESSIVENESS: std::cell::Cell<u8> = std::cell::Cell::new(5);
}

const INFINITY: i32 = 30000;
const MATE_SCORE: i32 = 29000;

// Thread-local counter to reduce atomic contention (batch updates to global)
thread_local!(static LOCAL_NODE_COUNTER: std::cell::Cell<u64> = std::cell::Cell::new(0));

// Epoch for clearing thread-local history
static HISTORY_EPOCH: AtomicU64 = AtomicU64::new(0);
thread_local!(static LOCAL_EPOCH: std::cell::Cell<u64> = std::cell::Cell::new(0));

static NODE_COUNT: AtomicU64 = AtomicU64::new(0);
static QNODE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static STOP_SEARCH: AtomicBool = AtomicBool::new(false);

pub fn stop() {
    STOP_SEARCH.store(true, Ordering::Relaxed);
}

pub fn clear_stop_flag() {
    STOP_SEARCH.store(false, Ordering::Relaxed);
}

#[derive(Clone, Copy)]
pub struct TimeManager {
    pub start_time: std::time::Instant,
    pub allocated_time: u128,
}

impl TimeManager {
    pub fn new(wtime: Option<u64>, btime: Option<u64>, movestogo: Option<u64>, turn: chess::Color) -> Option<Self> {
        let time_left = if turn == chess::Color::White { wtime } else { btime };
        
        if let Some(t) = time_left {
            let moves = movestogo.unwrap_or(30).max(1);
            let mut alloc = t / moves;
            alloc = alloc.saturating_sub(50);
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

pub fn reset_node_counts() {
    NODE_COUNT.store(0, Ordering::Relaxed);
    QNODE_COUNT.store(0, Ordering::Relaxed);
    STOP_SEARCH.store(false, Ordering::Relaxed);
}

pub fn get_node_counts() -> (u64, u64) {
    (NODE_COUNT.load(Ordering::Relaxed), QNODE_COUNT.load(Ordering::Relaxed))
}

/// Check and clear thread-local history if epoch changed
#[inline(always)]
fn check_and_clear_history() {
    let global_epoch = HISTORY_EPOCH.load(Ordering::Relaxed);
    LOCAL_EPOCH.with(|e| {
        if e.get() < global_epoch {
            // Clear History
            HISTORY.with(|h| {
                for i in 0..64 {
                    for j in 0..64 {
                        h.borrow_mut()[i][j] = 0;
                    }
                }
            });
            
            // Clear Killers
            KILLERS.with(|k| {
                *k.borrow_mut() = [[None; 2]; 64];
            });
            
            // Clear Counter History (Safe Heap Allocation)
            COUNTER_HISTORY.with(|ch| {
                // Defeat Stack Overflow: Allocate on heap first!
                let v = vec![[[[0; 64]; 64]; 64]; 64];
                *ch.borrow_mut() = v.into_boxed_slice().try_into().unwrap();
            });
            
            e.set(global_epoch);
        }
    });
}

#[inline(always)]
fn update_killers(ply: u8, mv: ChessMove) {
    let ply_idx = (ply as usize).min(63);
    KILLERS.with(|k| {
        let mut killers = k.borrow_mut();
        if killers[ply_idx][0] != Some(mv) {
            killers[ply_idx][1] = killers[ply_idx][0];
            killers[ply_idx][0] = Some(mv);
        }
    });
}

#[inline(always)]
fn update_history(mv: ChessMove, depth: u8) {
    let from = mv.get_source().to_index();
    let to = mv.get_dest().to_index();
    
    // Gravity / Decay Logic (Stockfish Style)
    // Bonus is capped to prevent overflow
    let bonus = ((depth as i32) * (depth as i32)).min(400); 

    HISTORY.with(|h| {
        let mut hist = h.borrow_mut();
        let old_score = hist[from][to];
        
        // "Gravity": New Score = Old + Bonus - (Old * |Bonus|) / Scale
        // This naturally decays the score if Bonus is 0 (though here always positive for good moves)
        // Wait, standard history update is ONLY positive for beta-cutoffs.
        // True "Gravity" requires penalizing other moves. that's intricate.
        // For now, let's implement "Variable Term Approach":
        // score += bonus - score * abs(bonus) / 16384
        
        // Constant 10000 limit is simplistic. Let's use decayed update.
        let term = bonus - (old_score * bonus.abs() / 512); // Reduced scale for faster adaptation
        
        hist[from][to] += term;
        
        // Soft clamp just in case
        if hist[from][to] > 10000 { hist[from][to] = 10000; }
        if hist[from][to] < -10000 { hist[from][to] = -10000; }
    });
}

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

fn quiescence(board: &Board, eval_state: eval::EvalState, mut alpha: i32, beta: i32, ply: u8) -> i32 {
    QNODE_COUNT.fetch_add(1, Ordering::Relaxed);
    
    // Pass existing state! O(1)
    let stand_pat = eval::evaluate_lazy(board, &eval_state, alpha, beta);
    
    if ply > 32 {
        return stand_pat;
    }
    
    if stand_pat >= beta {
        return beta;
    }
    if alpha < stand_pat {
        alpha = stand_pat;
    }
    
    let mut targets = *board.color_combined(!board.side_to_move());
    if let Some(ep_sq) = board.en_passant() {
        targets |= chess::BitBoard::from_square(ep_sq);
    }

    let mut gen = MoveGen::new_legal(board);
    gen.set_iterator_mask(targets);
    
    // Allocation-free MoveList
    let mut captures = crate::movegen::MoveList::new();
    
    // Qodo: Track promotions to defer delta pruning
    let mut has_promotions = false;
    
    for m in gen {
         // Skip non-tactical moves
         if !eval::is_tactical(board, m) { continue; }
         
         // SEE pruning
         if eval::see(board, m) < 0 { continue; }
         
         if m.get_promotion().is_some() {
             has_promotions = true;
         }

         // SEE pruning removed (eval::see was buggy)
         let score = eval::mvv_lva_score(board, m);
         captures.push(m, score);
    }
    
    // Qodo: Delta pruning AFTER checking for promotions
    // Don't prune if there are promotions (they can change material significantly)
    const DELTA: i32 = 2500; // ~Queen value
    if !has_promotions && stand_pat + DELTA < alpha {
        return alpha;
    }
    
    // Convert to mutable slice for in-place selection sort
    let list = captures.as_slice_mut();
    let len = list.len();
    
    for i in 0..len {
        // Selection sort: find best move in [i..len]
        let mut best_idx = i;
        let mut best_score = list[i].score;
        
        for j in (i + 1)..len {
            if list[j].score > best_score {
                best_score = list[j].score;
                best_idx = j;
            }
        }
        
        // Swap to front
        list.swap(i, best_idx);
        let mv = list[i].mv;
        
        // Incremental Update
        let mut new_eval = eval_state;
        new_eval.apply_move(board, mv);
        
        // Move processing
        let new_board = board.make_move_new(mv);
        let score = -quiescence(&new_board, new_eval, -beta, -alpha, ply + 1);
        
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }
    
    alpha
}

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
    if ply >= MAX_PLY { return (eval::evaluate(board), None); }

    // Optimized: Use thread-local counter to batch atomic updates
    let check_time = LOCAL_NODE_COUNTER.with(|c| {
        let count = c.get() + 1;
        c.set(count);
        if count & 2047 == 0 {
            // Batch update global stats
            NODE_COUNT.fetch_add(2048, Ordering::Relaxed);
            true
        } else {
            false
        }
    });

    if check_time {
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
        let q_score = quiescence(board, eval_state, alpha, beta, 0);
        return (q_score, None);
    }
    
    let hash = board.get_hash();
    let in_check = *board.checkers() != chess::EMPTY;
    let phase = eval_state.phase; // Use O(1) phase
    let is_endgame = phase < 12;
    
    let tt_result = tt_probe(hash);

    // Decode AND VALIDATE the TT move
    let tt_move: Option<ChessMove> = tt_result.as_ref()
        .and_then(|(e, _)| decode_move(e.best_move))
        .filter(|&mv| board.legal(mv)); 
    
    if let Some((entry, flag)) = tt_result {
        if entry.depth >= depth {
            let score = entry.score as i32;
            let is_valid_entry = tt_move.is_some() || (flag == TTFlag::Alpha);

            if is_valid_entry {
                match flag {
                    TTFlag::Exact => return (score, tt_move),
                    TTFlag::Alpha => if score <= alpha { return (alpha, tt_move); },
                    TTFlag::Beta => if score >= beta { return (beta, tt_move); },
                    TTFlag::None => {}
                }
            }
        }
    }
    
    let static_eval = if !in_check { eval::evaluate_with_state(board, &eval_state, alpha, beta) } else { 0 };
    
    if !in_check && depth <= 8 {
        let rfp_margin = 100 * depth as i32;
        if static_eval - rfp_margin >= beta {
            return (beta, None);
        }
    }
    
    // Razoring removed (Stockfish HCE audit) -- proved ineffective at modern depths
    // if !in_check && depth <= 3 && tt_move.is_none() { ... }
    
    // Null Move Pruning
    let us = board.side_to_move();
    let non_pawns = *board.color_combined(us) & !*board.pieces(chess::Piece::Pawn) & !*board.pieces(chess::Piece::King);
    let zugzwang_risk = non_pawns.0 == 0;

    if null_ok && !in_check && !is_endgame && !zugzwang_risk && depth >= 3 {
        // Dynamic R based on static_eval to be safer? No, standard 3 is fine for depth > 6.
        // let r = if depth > 6 { 3 } else { 2 };
        if let Some(null_board) = board.null_move() {
            // Adaptive NMP reduction (Stockfish formula: 3 + depth/6)
            let r = 3 + depth / 6;
            let (score, _) = negamax(&null_board, eval_state, None, depth.saturating_sub(1 + r), -beta, -beta + 1, ply + 1, false, time_manager);
            let score = -score;
            if STOP_SEARCH.load(Ordering::Relaxed) { return (0, None); }
            if score >= beta {
                return (beta, None);
            }
        }
    }
    
    if !in_check && depth >= 5 && tt_move.is_some() {
        let probcut_beta = beta + 200;
        let probcut_depth = depth - 4;

        let mut gen = MoveGen::new_legal(board);
        let targets = board.color_combined(!board.side_to_move());
        gen.set_iterator_mask(*targets);

        for mv in gen {
            // SEE pruning
            if eval::see(board, mv) < 0 { continue; }

            let mut new_eval = eval_state;
            new_eval.apply_move(board, mv);
            let new_board = board.make_move_new(mv);

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
    
    let mut tt_move = tt_move;
    if tt_move.is_none() && depth >= 6 {
         let iid_depth = depth - 2;
         let _ = negamax(board, eval_state, prev_move, iid_depth, alpha, beta, ply, null_ok, time_manager);
         if let Some((entry, _)) = tt_probe(hash) {
             // Code Rabbit Fix: Validate IID TT move legality
             tt_move = decode_move(entry.best_move).filter(|&mv| board.legal(mv));
         }
    }

    let mut extension: u8 = 0;
    if !in_check && depth >= 8 {
        if let Some((ref entry, TTFlag::Beta)) = tt_result {
            if let Some(tm) = tt_move {
                if entry.depth >= depth.saturating_sub(3) {
                    let se_beta = (entry.score as i32).saturating_sub(2 * depth as i32);
                    let se_depth = (depth - 1) / 2;

                    let mut is_singular = true;
                    let mut moves_checked = 0;
                    let max_checks = 6;

                    let mut se_gen = StagedMoveGen::new(board, None, ply);

                    while let Some(other_mv) = se_gen.next() {
                        if other_mv == tm { continue; }

                        moves_checked += 1;
                        if moves_checked > max_checks {
                            is_singular = false;
                            break;
                        }

                        let mut new_eval = eval_state;
                        new_eval.apply_move(board, other_mv);
                        let new_board = board.make_move_new(other_mv);

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
        
        let mut new_eval = eval_state;
        new_eval.apply_move(board, mv);
        let new_board = board.make_move_new(mv);
        let gives_check = *new_board.checkers() != chess::EMPTY;
        
        // LMP: Late Move Pruning
        // Formula: count > (24 + depth * depth) (Relaxed from 16)
        let lmp_threshold = (24 + depth as usize * depth as usize).min(63);
        if depth <= 5 && !in_check && !eval::is_tactical(board, mv) && !gives_check && i >= lmp_threshold && i > 0 {
            continue;
        }
        
        if depth <= 4 && !in_check && !eval::is_tactical(board, mv) && !gives_check && i > 0 {
            let margin = 200 * depth as i32;
            if static_eval + margin < alpha {
                continue;
            }
        }
        
        let mut reduction = 0u8;
        if i >= 2 && depth >= 4 && !eval::is_tactical(board, mv) && !gives_check && !in_check {
             let ply_idx = (ply as usize).min(63);
             let is_killer = KILLERS.with(|k| {
                 let k = k.borrow();
                 k[ply_idx][0] == Some(mv) || k[ply_idx][1] == Some(mv)
             });

             if !is_killer {
                 // Logarithmic Reduction
                 let mut r = 1.0 + (depth as f32).ln() * (i as f32).ln() / 2.6;

                 // History-based LMR reduction
                 // Good history -> reduce less. Bad history -> reduce more.
                 let hist_score = HISTORY.with(|h| {
                     let h = h.borrow();
                     let from = mv.get_source().to_index();
                     let to = mv.get_dest().to_index();
                     h[from][to]
                 });
                 
                 // Reduce less if history is high (bonus). Re-add if history is low (malus).
                 // Tune: Dampened effect (divide by 8192.0 instead of 2048) to be safer
                 r -= (hist_score as f32) / 8192.0; 
                 
                 reduction = r.clamp(0.0, depth as f32) as u8;
             }
        }
        
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

/// Helper function to reset thread-local stats at start of search
fn reset_thread_local_stats() {
    KILLERS.with(|k| *k.borrow_mut() = [[None; 2]; 64]);
    HISTORY.with(|h| {
        let mut hist = h.borrow_mut();
        for i in 0..64 {
            for j in 0..64 {
                hist[i][j] /= 2;
            }
        }
    });
}

/// The core search loop used by all threads
fn iterative_deepening_worker(
    board: &Board,
    max_depth: u8,
    aggressiveness: u8,
    time_manager: Option<TimeManager>,
    is_main_thread: bool
) -> (Option<ChessMove>, i32) {
    check_and_clear_history();
    AGGRESSIVENESS.with(|a| a.set(aggressiveness));
    
    reset_thread_local_stats();
    
    let mut best_move = None;
    let mut best_score = 0;
    let root_eval = eval::EvalState::new(board);
    
    for depth in 1..=max_depth {
        // Sourcery Fix: Helper threads must check stop flag at top of loop
        if STOP_SEARCH.load(Ordering::Relaxed) {
            break;
        }

        if let Some(tm) = time_manager {
            // Sourcery Fix: Only main thread should apply heuristic early cutoff
            if is_main_thread && depth > 4 && tm.start_time.elapsed().as_millis() > tm.allocated_time / 2 {
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
        
        // Only main thread (implied by caller context, or just let all update best)
        // Actually, main thread should probably be the one returning the result.
        // But for Lazy SMP, all threads populate TT.
    }
    
    // Fallback
    if best_move.is_none() {
        best_move = MoveGen::new_legal(board).next();
    }

    (best_move, best_score)
}

/// Standard Serial Search (wrapper)
pub fn iterative_deepening(
    board: &Board, 
    max_depth: u8, 
    aggressiveness: u8,
    wtime: Option<u64>,
    btime: Option<u64>,
    movestogo: Option<u64>
) -> (Option<ChessMove>, i32) {
    
    // Check Tablebase first!
    if let Some((tb_move, score)) = tablebase::probe_root(board) {
        if is_debug() {
             eprintln!("[DEBUG] Syzygy hit: {} score {}", tb_move, score);
        }
        return (Some(tb_move), score);
    }

    reset_node_counts();
    
    let time_manager = TimeManager::new(wtime, btime, movestogo, board.side_to_move());
    iterative_deepening_worker(board, max_depth, aggressiveness, time_manager, true)
}

/// Lazy SMP Search - Spawns helper threads to spam the search
pub fn lazy_smp_search(
    board: &Board,
    max_depth: u8,
    aggressiveness: u8,
    wtime: Option<u64>,
    btime: Option<u64>,
    movestogo: Option<u64>,
    num_threads: usize
) -> (Option<ChessMove>, i32) {
    // Sourcery Fix: Clamp thread count to guard against misuse
    let num_threads = num_threads.clamp(1, 64);

    if num_threads <= 1 {
        return iterative_deepening(board, max_depth, aggressiveness, wtime, btime, movestogo);
    }
    
    // Check Tablebase first!
    if let Some((tb_move, score)) = tablebase::probe_root(board) {
         return (Some(tb_move), score);
    }

    reset_node_counts();
    let time_manager = TimeManager::new(wtime, btime, movestogo, board.side_to_move());
    
    // Spawn helper threads
    let mut handles = Vec::new();
    let helper_board = board.clone();
    
    for _ in 0..num_threads-1 {
        let b = helper_board.clone();
        let tm = time_manager.clone();
        handles.push(thread::spawn(move || {
            // Helpers execute search but ignore result, they just populate TT
            iterative_deepening_worker(&b, max_depth, aggressiveness, tm, false);
        }));
    }
    
    // Main thread search
    let (best_move, score) = iterative_deepening_worker(board, max_depth, aggressiveness, time_manager, true);

    // Stop helpers
    stop();

    for h in handles {
        let _ = h.join();
    }

    // Clear stop flag for next search
    clear_stop_flag();

    (best_move, score)
}
