pub mod search;
pub mod eval;
mod book;
pub mod movegen;
pub mod tablebase;
pub mod pst;


use chess::{Board, MoveGen};
use std::str::FromStr;
use std::sync::OnceLock;
use pyo3::prelude::*;

// Global book instance (loaded once)
static BOOK: OnceLock<Option<book::OpeningBook>> = OnceLock::new();

// C-API Helpers
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

#[no_mangle]
pub extern "C" fn get_best_move_c(fen_ptr: *const c_char, depth: c_int, threads: c_int) -> *mut c_char {
    if fen_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let fen_c = unsafe { CStr::from_ptr(fen_ptr) };
    let fen = match fen_c.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    let board = match Board::from_str(fen) {
        Ok(b) => b,
        Err(_) => return std::ptr::null_mut(),
    };
    
    // Check book logic (simplified for C API)
    if let Some(book) = get_book() {
        if let Some(mv) = book.get_move(&board) {
            let s = CString::new(mv.to_string()).unwrap();
            return s.into_raw();
        }
    }
    
    // Search
    let use_parallel = threads > 1;
    let aggr = 5; // Default aggressiveness
    let thread_count = if use_parallel { threads as usize } else { 1 };
    
    let (best_move, _) = if use_parallel && depth >= 6 {
         search::lazy_smp_search(&board, depth as u8, aggr, None, None, None, thread_count)
    } else {
         search::iterative_deepening(&board, depth as u8, aggr, None, None, None)
    };
    
    let mv_str = match best_move {
        Some(mv) => mv.to_string(),
        None => "0000".to_string(),
    };
    
    let s = CString::new(mv_str).unwrap();
    s.into_raw()
}

#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    if s.is_null() { return; }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

#[no_mangle]
pub extern "C" fn get_node_counts_c(nodes: *mut u64, qnodes: *mut u64) {
    let (n, q) = search::get_node_counts();
    unsafe {
        if !nodes.is_null() { *nodes = n; }
        if !qnodes.is_null() { *qnodes = q; }
    }
}

#[no_mangle]
pub extern "C" fn clear_tt_c() {
    search::tt_clear();
}

#[no_mangle]
pub extern "C" fn evaluate_c(fen_ptr: *const c_char) -> c_int {
    let fen_c = unsafe { CStr::from_ptr(fen_ptr) };
    let fen = match fen_c.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    
    let board = match Board::from_str(fen) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    
    eval::evaluate(&board)
}

fn get_book() -> &'static Option<book::OpeningBook> {
    BOOK.get_or_init(|| book::load_best_book())
}

/// Get the best move for a given FEN position using alpha-beta search
/// First checks opening book, then falls back to search
/// aggressiveness: 1-10, higher = more pruning = faster but riskier
/// use_parallel: if true, use multi-threaded search (faster on multi-core)
/// wtime/btime: time left in ms
/// movestogo: moves to go
#[pyfunction]
#[pyo3(signature = (fen, depth, aggressiveness=5, use_parallel=false, wtime=None, btime=None, movestogo=None))]
fn get_best_move(
    fen: &str, 
    depth: u8, 
    aggressiveness: u8, 
    use_parallel: bool,
    wtime: Option<u64>,
    btime: Option<u64>,
    movestogo: Option<u64>
) -> PyResult<String> {
    let board = Board::from_str(fen).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid FEN: {}", e))
    })?;
    
    // Check opening book first
    if let Some(book) = get_book() {
        if let Some(book_move) = book.get_move(&board) {
            return Ok(book_move.to_string());
        }
    }
    
    // Fall back to search with aggressiveness setting
    let aggr = aggressiveness.clamp(1, 10);
    
    let (best_move, _score) = if use_parallel && depth >= 6 {
        // Use parallel search for higher depths - detect cores automatically
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(64); // Cap at 64 threads to match UCI limit

        search::lazy_smp_search(&board, depth, aggr, wtime, btime, movestogo, threads)
    } else {
        // Use regular iterative deepening
        search::iterative_deepening(&board, depth, aggr, wtime, btime, movestogo)
    };
    
    match best_move {
        Some(mv) => Ok(mv.to_string()),
        None => Ok("0000".to_string()),
    }
}

/// Get list of legal moves for a position
#[pyfunction]
fn get_legal_moves(fen: &str) -> PyResult<Vec<String>> {
    let board = Board::from_str(fen).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid FEN: {}", e))
    })?;
    
    let moves: Vec<String> = MoveGen::new_legal(&board)
        .map(|m| m.to_string())
        .collect();
    
    Ok(moves)
}

/// Evaluate a position (positive = white is better)
#[pyfunction]
fn evaluate(fen: &str) -> PyResult<i32> {
    let board = Board::from_str(fen).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid FEN: {}", e))
    })?;
    
    Ok(eval::evaluate(&board))
}

/// Debug: Get breakdown of evaluation terms
#[pyfunction]
fn debug_eval_components(fen: &str) -> PyResult<std::collections::HashMap<String, i32>> {
    let board = Board::from_str(fen).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid FEN: {}", e))
    })?;
    
    Ok(eval::debug_eval(&board))
}

/// Check if a position is in the opening book
#[pyfunction]
fn has_book_move(fen: &str) -> PyResult<bool> {
    let board = Board::from_str(fen).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid FEN: {}", e))
    })?;
    
    if let Some(book) = get_book() {
        return Ok(book.get_move(&board).is_some());
    }
    Ok(false)
}

/// Get node counts from last search (nodes, qnodes)
#[pyfunction]
fn get_node_counts() -> PyResult<(u64, u64)> {
    Ok(search::get_node_counts())
}

/// Clear the transposition table - MUST call before new games/benchmarks!
/// This removes poisoned TT entries from previous crashed/failed searches
#[pyfunction]
fn clear_tt() -> PyResult<()> {
    search::tt_clear();
    Ok(())
}

/// Enable or disable debug mode for verbose logging
#[pyfunction]
fn set_debug(enabled: bool) -> PyResult<()> {
    search::set_debug(enabled);
    Ok(())
}

/// Set the Syzygy tablebase path
#[pyfunction]
fn set_tablebase_path(path: &str) -> PyResult<()> {
    let _ = tablebase::init(path);
    Ok(())
}

/// Stop search helper for Python (optional, but good for completeness)
#[pyfunction]
fn stop_search() -> PyResult<()> {
    search::stop();
    Ok(())
}

/// Python module
#[pymodule]
fn rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_best_move, m)?)?;
    m.add_function(wrap_pyfunction!(get_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(debug_eval_components, m)?)?;
    m.add_function(wrap_pyfunction!(has_book_move, m)?)?;
    m.add_function(wrap_pyfunction!(get_node_counts, m)?)?;
    m.add_function(wrap_pyfunction!(clear_tt, m)?)?;
    m.add_function(wrap_pyfunction!(set_debug, m)?)?;
    m.add_function(wrap_pyfunction!(set_tablebase_path, m)?)?;
    m.add_function(wrap_pyfunction!(stop_search, m)?)?;
    Ok(())
}
