use std::io::{self, BufRead};
use rust_engine::search;
use chess::Board;
use std::str::FromStr;
use std::thread::{self, JoinHandle};

fn main() {
    let stdin = io::stdin();
    let mut board = Board::default();
    let mut active_search_handle: Option<JoinHandle<()>> = None;

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
                println!("option name SyzygyPath type string default <empty>");
                println!("uciok");
            },
            "setoption" => {
                if commands.len() >= 5 && commands[1] == "name" && commands[3] == "value" {
                    let name = commands[2];
                    let value = commands[4..].join(" ");
                    match name.to_lowercase().as_str() {
                        "hash" => {
                            if let Ok(_v) = value.parse::<usize>() {
                                // TODO: Resize Transposition Table
                            }
                        },
                        "threads" => {
                            if let Ok(_v) = value.parse::<usize>() {
                                // TODO: Set thread pool size
                            }
                        },
                        "syzygypath" => {
                             if let Err(e) = rust_engine::tablebase::init(&value) {
                                 println!("info string {}", e);
                             }
                        },
                        _ => {
                            println!("info string Option {} not supported", name);
                        }
                    }
                } else {
                     println!("info string Invalid setoption command format");
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
                        if !found {
                            eprintln!("info string Invalid move: {}", target);
                        }
                    }
                }
            },
            "go" => {
                // Stop and join any active search
                search::stop();
                if let Some(handle) = active_search_handle.take() {
                    let _ = handle.join();
                }

                // Reset stop flag for new search
                search::clear_stop_flag();
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
                active_search_handle = Some(thread::spawn(move || {
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
                }));
            },
            "stop" => {
                search::stop();
                if let Some(handle) = active_search_handle.take() {
                    let _ = handle.join();
                }
            },
            "quit" => {
                search::stop();
                if let Some(handle) = active_search_handle.take() {
                    let _ = handle.join();
                }
                break;
            },
            _ => {}
        }
    }
}
