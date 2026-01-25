use std::io::{self, BufRead};
use rust_engine::{search, eval};
use chess::{Board, ChessMove, Color};
use std::str::FromStr;
use std::thread;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

fn main() {
    let stdin = io::stdin();
    let mut board = Board::default();

    // UCI Loop
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let commands: Vec<&str> = line.split_whitespace().collect();

        if commands.is_empty() { continue; }

        match commands[0] {
            "uci" => {
                println!("id name RustEngine");
                println!("id author Hiep & Jules");
                println!("option name Hash type spin default 64 min 1 max 1024");
                println!("option name Threads type spin default 1 min 1 max 64");
                println!("uciok");
            },
            "setoption" => {
                // setoption name Hash value 128
                // Basic parsing
                if commands.len() >= 5 && commands[1] == "name" && commands[3] == "value" {
                    let name = commands[2];
                    let value = commands[4];
                    match name.to_lowercase().as_str() {
                        "hash" => {
                            // Logic to resize TT would go here
                            // For now we acknowledge it
                            if let Ok(_v) = value.parse::<usize>() {
                                // TODO: Resize TT
                            }
                        },
                        "threads" => {
                            // Logic to set thread pool size
                            if let Ok(_v) = value.parse::<usize>() {
                                // TODO: Set threads
                            }
                        },
                        _ => {}
                    }
                }
            },
            "isready" => println!("readyok"),
            "ucinewgame" => {
                search::tt_clear();
            },
            "position" => {
                if commands.len() < 2 { continue; }
                // position startpos moves e2e4 ...
                // position fen ... moves ...
                let mut moves_idx = 1;
                if commands[1] == "startpos" {
                    board = Board::default();
                    moves_idx = 2;
                } else if commands[1] == "fen" {
                    let mut fen_parts = Vec::new();
                    moves_idx = 2;
                    while moves_idx < commands.len() && commands[moves_idx] != "moves" {
                        fen_parts.push(commands[moves_idx]);
                        moves_idx += 1;
                    }
                    let fen = fen_parts.join(" ");
                    if let Ok(b) = Board::from_str(&fen) {
                        board = b;
                    }
                }

                if moves_idx < commands.len() && commands[moves_idx] == "moves" {
                    for m_str in &commands[moves_idx+1..] {
                        let move_gen = chess::MoveGen::new_legal(&board);
                        // Find matching move
                        // This is a bit inefficient for parsing but fine for UCI
                        // We need to parse coordinate notation to ChessMove
                        // The 'chess' crate doesn't have a direct 'from_san' or 'from_uci' easily accessible
                        // that handles promotion without some work, but let's try a simple iteration search.

                        // We need to construct a move from the string (e.g., "e2e4", "a7a8q")
                        // For now, let's assume we can match it against legal moves strings.
                        let target = m_str.to_string();
                        let mut found = false;
                        for m in move_gen {
                             if m.to_string() == target {
                                 board = board.make_move_new(m);
                                 found = true;
                                 break;
                             }
                        }
                    }
                }
            },
            "go" => {
                // Reset stop flag
                search::reset_node_counts();

                // Parse time controls
                let mut wtime = None;
                let mut btime = None;
                let mut movestogo = None;
                let mut depth = 64; // Default max

                let mut i = 1;
                while i < commands.len() {
                    match commands[i] {
                        "wtime" => { if i+1 < commands.len() { wtime = commands[i+1].parse().ok(); } },
                        "btime" => { if i+1 < commands.len() { btime = commands[i+1].parse().ok(); } },
                        "movestogo" => { if i+1 < commands.len() { movestogo = commands[i+1].parse().ok(); } },
                        "depth" => { if i+1 < commands.len() { depth = commands[i+1].parse().unwrap_or(64); } },
                        _ => {}
                    }
                    i += 1;
                }

                let search_board = board.clone();

                // Spawn search thread
                thread::spawn(move || {
                    let (best_move, _) = search::iterative_deepening(
                        &search_board,
                        depth,
                        5, // default aggressiveness
                        wtime,
                        btime,
                        movestogo
                    );

                    if let Some(mv) = best_move {
                        println!("bestmove {}", mv);
                    } else {
                        println!("bestmove 0000");
                    }
                });
            },
            "stop" => {
                // Signal search to stop
                // We use the exposed search::set_debug method to access internals or just rely on atomic
                // But wait, STOP_SEARCH is private in search.rs.
                // We need a public method to set it.
                // Let's assume we add `search::stop()` in search.rs or use existing mechanism.
                // The plan said "uci.rs can set the rust_engine::search::STOP_SEARCH directly if exposed".
                // If not exposed, I need to expose it.
                // For now, I will use a placeholder and fix search.rs in next step if needed.
                // Actually, I can write to search.rs to make it public or add a helper.
                search::stop();
            },
            "quit" => break,
            _ => {}
        }
    }
}
