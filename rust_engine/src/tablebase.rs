use std::path::Path;
use shakmaty_syzygy::{Tablebase, AmbiguousWdl}; 
use shakmaty::{Chess, Position, CastlingMode, Move};
use chess::{Board, ChessMove, Square};
use std::sync::OnceLock;
use std::str::FromStr;

// Global Tablebase instance
// Check tablebases.rs lines 2-4 for imports.
// OnceLock is safe for initialization.
static TABLEBASE: OnceLock<Tablebase<Chess>> = OnceLock::new();

/// Initialize the tablebase with a path
pub fn init(path: &str) -> Result<(), String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("Syzygy path does not exist: {}", path));
    }

    let mut tb = Tablebase::new();
    match tb.add_directory(p) {
        Ok(_) => {
            TABLEBASE.set(tb).map_err(|_| "Tablebase already initialized".to_string())?;
            println!("info string Syzygy tablebases found at: {}", path);
            Ok(())
        },
        Err(e) => Err(format!("Failed to load Syzygy: {}", e))
    }
}

/// Convert chess::Board to shakmaty::Chess
fn convert_board(board: &Board) -> Option<Chess> {
    let fen = format!("{}", board);
    use shakmaty::fen::Fen;
    
    let setup: Fen = Fen::from_str(&fen).ok()?;
    setup.into_position(CastlingMode::Standard).ok()
}

/// Convert shakmaty::Move to chess::ChessMove
fn convert_move(m: &Move, _pos: &Chess) -> Option<ChessMove> {
    let from = m.from()?;
    let to = m.to();
    let prom = m.promotion(); 
    
    let from_sq = Square::from_str(&from.to_string()).ok()?;
    let to_sq = Square::from_str(&to.to_string()).ok()?;
    
    let prom_piece = match prom {
        Some(role) => match role {
            shakmaty::Role::Queen => Some(chess::Piece::Queen),
            shakmaty::Role::Rook => Some(chess::Piece::Rook),
            shakmaty::Role::Bishop => Some(chess::Piece::Bishop),
            shakmaty::Role::Knight => Some(chess::Piece::Knight),
            _ => None,
        },
        None => None,
    };
    
    Some(ChessMove::new(from_sq, to_sq, prom_piece))
}

/// Probe for WDL (Win/Draw/Loss) score
pub fn probe_wdl(board: &Board) -> Option<i32> {
    if let Some(tb) = TABLEBASE.get() {
        let pos = convert_board(board)?;
        if pos.board().occupied().count() > 7 { 
            return None;
        }

        match tb.probe_wdl(&pos) {
             Ok(wdl) => {
                 match wdl {
                     AmbiguousWdl::Win => Some(20000),
                     AmbiguousWdl::Loss => Some(-20000),
                     AmbiguousWdl::Draw => Some(0),
                     AmbiguousWdl::CursedWin => Some(10000), 
                     AmbiguousWdl::BlessedLoss => Some(-10000),
                     _ => Some(0), 
                 }
             },
             _ => None
        }
    } else {
        None
    }
}

/// Probe for Best Move (Root)
/// Returns (BestMove, Score)
pub fn probe_root(board: &Board) -> Option<(ChessMove, i32)> {
    let tb = TABLEBASE.get()?;
    let pos = convert_board(board)?;
    if pos.board().occupied().count() > 7 {
        return None;
    }

    let mut best_move: Option<ChessMove> = None;
    let mut best_score = -30000;
    
    let moves = pos.legal_moves();
    
    for m in moves {
         let mut new_pos = pos.clone();
         new_pos.play_unchecked(m.clone());
         
         if let Ok(wdl) = tb.probe_wdl(&new_pos) {
             let score_val = match wdl {
                 AmbiguousWdl::Win => -20000,   // They win -> We lose
                 AmbiguousWdl::Loss => 20000,   // They lose -> We win
                 AmbiguousWdl::Draw => 0,
                 AmbiguousWdl::CursedWin => -10000,
                 AmbiguousWdl::BlessedLoss => 10000,
                 _ => 0,
             };
             
             if score_val > best_score {
                 best_score = score_val;
                 // Pass `pos` for correct UCI conversion context if needed?
                 // Wait, `convert_move` needs the *source* position (the one BEFORE the move).
                 // So we pass `&pos`.
                 if let Some(cm) = convert_move(&m, &pos) {
                     best_move = Some(cm);
                 }
             }
         }
    }
    
    if let Some(mv) = best_move {
        return Some((mv, best_score));
    }
    None
}
