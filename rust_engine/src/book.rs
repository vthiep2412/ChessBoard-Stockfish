//! Polyglot Opening Book Reader
//! Format: https://hardy.uhasselt.be/Toga/book_format.html

use chess::{Board, ChessMove, Square, Piece, File, Rank};
use std::fs::File as FsFile;
use std::io::{BufReader, Read};
use std::path::Path;

/// Polyglot book entry (16 bytes each)
#[derive(Debug, Clone, Copy)]
pub struct BookEntry {
    pub key: u64,
    pub mv: u16,
    pub weight: u16,
    pub _learn: u32,
}

/// Opening book reader
pub struct OpeningBook {
    entries: Vec<BookEntry>,
}

impl OpeningBook {
    /// Load a Polyglot book from file
    pub fn load(path: &Path) -> Option<Self> {
        let file = FsFile::open(path).ok()?;
        let file_size = file.metadata().ok()?.len();
        
        // Validate file size is a multiple of 16 bytes (each entry is 16 bytes)
        if file_size % 16 != 0 {
            eprintln!("Warning: Book file size {} is not a multiple of 16, may be corrupt", file_size);
            return None;
        }
        
        let entry_count = file_size / 16;
        
        let mut reader = BufReader::new(file);
        let mut entries = Vec::with_capacity(entry_count as usize);
        
        for _ in 0..entry_count {
            let mut buf = [0u8; 16];
            if reader.read_exact(&mut buf).is_err() {
                eprintln!("Error: Failed to read complete book entry");
                return None;
            }
            
            // Use slice-to-array conversion for cleaner parsing
            let entry = BookEntry {
                key: u64::from_be_bytes(buf[0..8].try_into().unwrap()),
                mv: u16::from_be_bytes(buf[8..10].try_into().unwrap()),
                weight: u16::from_be_bytes(buf[10..12].try_into().unwrap()),
                _learn: u32::from_be_bytes(buf[12..16].try_into().unwrap()),
            };
            
            entries.push(entry);
        }
        
        // Sort by key for binary search
        entries.sort_by_key(|e| e.key);
        
        Some(OpeningBook { entries })
    }
    
    /// Get a move from the book for the given position
    pub fn get_move(&self, board: &Board) -> Option<ChessMove> {
        let key = polyglot_hash(board);
        
        // Find all entries matching this position
        let mut matches: Vec<&BookEntry> = Vec::new();
        
        // Binary search for first match
        let idx = self.entries.binary_search_by_key(&key, |e| e.key);
        if let Ok(start) = idx {
            // Collect all entries with same key
            let mut i = start;
            while i > 0 && self.entries[i - 1].key == key {
                i -= 1;
            }
            while i < self.entries.len() && self.entries[i].key == key {
                matches.push(&self.entries[i]);
                i += 1;
            }
        }
        
        if matches.is_empty() {
            return None;
        }
        
        // Pick best move (highest weight)
        let best = matches.iter().max_by_key(|e| e.weight)?;
        decode_polyglot_move(board, best.mv)
    }
}

/// Convert Polyglot move encoding to ChessMove
fn decode_polyglot_move(board: &Board, mv: u16) -> Option<ChessMove> {
    let to_file = (mv & 0x7) as usize;
    let to_rank = ((mv >> 3) & 0x7) as usize;
    let from_file = ((mv >> 6) & 0x7) as usize;
    let from_rank = ((mv >> 9) & 0x7) as usize;
    let promo = ((mv >> 12) & 0x7) as usize;
    
    let files = [File::A, File::B, File::C, File::D, File::E, File::F, File::G, File::H];
    let ranks = [Rank::First, Rank::Second, Rank::Third, Rank::Fourth, 
                 Rank::Fifth, Rank::Sixth, Rank::Seventh, Rank::Eighth];
    
    let from_sq = Square::make_square(ranks[from_rank], files[from_file]);
    let to_sq = Square::make_square(ranks[to_rank], files[to_file]);
    
    let promotion = match promo {
        1 => Some(Piece::Knight),
        2 => Some(Piece::Bishop),
        3 => Some(Piece::Rook),
        4 => Some(Piece::Queen),
        _ => None,
    };
    
    // Find matching legal move
    for legal_move in chess::MoveGen::new_legal(board) {
        if legal_move.get_source() == from_sq && legal_move.get_dest() == to_sq {
            if promotion.is_none() || legal_move.get_promotion() == promotion {
                return Some(legal_move);
            }
        }
    }
    
    None
}

/// Calculate Polyglot hash for a position
/// Uses shakmaty which has Polyglot-compatible Zobrist keys
fn polyglot_hash(board: &chess::Board) -> u64 {
    use shakmaty::{Chess, fen::Fen, zobrist::{Zobrist64, ZobristHash}};
    
    // Convert chess::Board to shakmaty::Chess via FEN
    let fen_str = format!("{}", board);
    
    if let Ok(fen) = fen_str.parse::<Fen>() {
        if let Ok(pos) = fen.into_position::<Chess>(shakmaty::CastlingMode::Standard) {
            // Get Polyglot-compatible hash
            let hash: Zobrist64 = pos.zobrist_hash(shakmaty::EnPassantMode::Legal);
            return hash.0;
        }
    }
    
    // Fallback to chess crate hash (won't match book, but won't crash)
    board.get_hash()
}

/// Try to load book from multiple paths
pub fn load_best_book() -> Option<OpeningBook> {
    let book_paths = [
        "books/komodo.bin",      // 9 MB, solid choice
        "books/cerebellum.bin",
        "books/perfect2023.bin", 
        "books/titans.bin",
        "books/gm2001.bin",
        "books/opening.bin",
        "../books/komodo.bin",
        "../books/cerebellum.bin",
        "../books/titans.bin",
    ];
    
    for path in &book_paths {
        if let Some(book) = OpeningBook::load(Path::new(path)) {
            // Validate book has entries (not empty/corrupt)
            if book.entries.len() > 100 {
                return Some(book);
            }
        }
    }
    
    None
}
