use pyo3::prelude::*;
use std::sync::OnceLock;

mod search;
mod eval;
mod book;

use chess::{Board, MoveGen};
use std::str::FromStr;

// Global book instance (loaded once)
static BOOK: OnceLock<Option<book::OpeningBook>> = OnceLock::new();

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
    
    // NOTE: Don't clear TT here! TT is essential for iterative deepening.
    // TT should only be cleared at game start (via new_game/tt_clear_all).
    
    // Fall back to search with aggressiveness setting
    let aggr = aggressiveness.clamp(1, 10);
    
    let (best_move, _score) = if use_parallel && depth >= 6 {
        // Use parallel search for higher depths
        search::parallel_root_search(&board, depth, aggr, wtime, btime, movestogo)
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

/// Python module
#[pymodule]
fn rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_best_move, m)?)?;
    m.add_function(wrap_pyfunction!(get_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(has_book_move, m)?)?;
    m.add_function(wrap_pyfunction!(get_node_counts, m)?)?;
    m.add_function(wrap_pyfunction!(clear_tt, m)?)?;
    m.add_function(wrap_pyfunction!(set_debug, m)?)?;
    Ok(())
}