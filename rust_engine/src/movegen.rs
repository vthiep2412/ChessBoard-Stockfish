use chess::{Board, ChessMove, MoveGen};
use crate::eval;
use crate::search::{KILLERS, HISTORY};

// Stage constants
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    TTMove,
    CapturesWinning,
    CapturesLosing,
    Killers,
    Quiets,
    BadQuiets, // Quiets that are not killers or good history
    Done,
}

pub struct StagedMoveGen {
    board: Board,
    tt_move: Option<ChessMove>,
    killers: [Option<ChessMove>; 2],
    history_ptr: *const [[i32; 64]; 64], // Raw pointer to history to avoid borrow checker issues? No, let's use RefCell access
    stage: Stage,
    captures_buffer: Vec<ChessMove>,
    quiets_buffer: Vec<ChessMove>,
    idx: usize,
}

impl StagedMoveGen {
    pub fn new(board: &Board, tt_move: Option<ChessMove>, ply: u8) -> Self {
        // Fetch killers for this ply
        // Explicitly type closure argument to satisfy compiler
        let killers = KILLERS.with(|k: &std::cell::RefCell<[[Option<ChessMove>; 2]; 64]>|
            k.borrow()[(ply as usize).min(63)]
        );

        Self {
            board: *board,
            tt_move,
            killers,
            history_ptr: std::ptr::null(), // Placeholder
            stage: Stage::TTMove,
            captures_buffer: Vec::with_capacity(16),
            quiets_buffer: Vec::with_capacity(32),
            idx: 0,
        }
    }

    // Generate captures and populate buffer
    fn generate_captures(&mut self) {
        let targets = self.board.color_combined(!self.board.side_to_move());
        let mut gen = MoveGen::new_legal(&self.board);
        gen.set_iterator_mask(*targets);

        // Add EP capture if available
        if let Some(ep_sq) = self.board.en_passant() {
            // Note: board.en_passant() returns the victim square? No, standard chess logic says EP square is destination.
            // Wait, chess crate en_passant() returns Option<Square>.
            // "The en passant target square is the square that the pawn passed over." (Wikipedia)
            // But usually engines store the square *behind* the pawn.
            // chess crate docs: "Returns the en passant square if one exists."
            // In FEN "e3", e3 is the destination.
            // Let's assume ep_sq IS the destination (which is empty).
            // So we just add it to the mask.
            // Using OR (|) instead of XOR (^) to be safe.

            let mask = *targets | chess::BitBoard::from_square(ep_sq);
            gen.set_iterator_mask(mask);
        }

        // Collect and score
        for m in gen {
             if Some(m) == self.tt_move { continue; }
             self.captures_buffer.push(m);
        }

        // Sort by MVV-LVA + SEE
        // Optimization: Sort only when needed (CapturesWinning vs Losing)
        // For now, simple sort
        let board = &self.board;
        self.captures_buffer.sort_by_key(|m| {
            let see = eval::see(board, *m);
            // Higher is better
            -(eval::mvv_lva_score(board, *m) + see * 10)
        });
    }

    // Generate quiets
    fn generate_quiets(&mut self) {
        let targets = !self.board.color_combined(!self.board.side_to_move()); // Non-captures
        let mut gen = MoveGen::new_legal(&self.board);
        gen.set_iterator_mask(targets); // Only quiets

        for m in gen {
            if Some(m) == self.tt_move { continue; }
            // Skip EP captures (they are in the quiet mask because dest is empty, but are captures)
            if Some(m.get_dest()) == self.board.en_passant() { continue; }
            self.quiets_buffer.push(m);
        }

        // Sort by History
        // Use thread-local access
        // Optimization: Pre-fetch history scores
        // We can't easily access thread local in sort closure without overhead
        // So we extract scores first

        let mut scored_quiets: Vec<(ChessMove, i32)> = self.quiets_buffer.iter().map(|m| {
            let from = m.get_source().to_index();
            let to = m.get_dest().to_index();
            // Explicit type for closure argument
            let score = HISTORY.with(|h: &std::cell::RefCell<[[i32; 64]; 64]>|
                h.borrow()[from][to]
            );
            (*m, score)
        }).collect();

        scored_quiets.sort_by_key(|(_, s)| -s);
        self.quiets_buffer = scored_quiets.into_iter().map(|(m, _)| m).collect();
    }
}

impl Iterator for StagedMoveGen {
    type Item = ChessMove;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::CapturesWinning;
                    if let Some(mv) = self.tt_move {
                        // Verify legality (TT move might be from collision)
                        // This is slow, maybe verify cheaply?
                        if chess::MoveGen::new_legal(&self.board).any(|m| m == mv) {
                             return Some(mv);
                        }
                    }
                },
                Stage::CapturesWinning => {
                    // Generate once
                    if self.captures_buffer.is_empty() && self.idx == 0 {
                        self.generate_captures();
                    }

                    if self.idx < self.captures_buffer.len() {
                        let mv = self.captures_buffer[self.idx];
                        self.idx += 1;
                        return Some(mv);
                    }

                    // Done with captures (both winning and losing are in buffer sorted)
                    self.stage = Stage::CapturesLosing;
                    self.idx = 0;
                },
                Stage::CapturesLosing => {
                     // Already handled in CapturesWinning in this simplified flow
                     self.stage = Stage::Killers;
                     self.idx = 0;
                },
                Stage::Killers => {
                     let k_idx = self.idx;
                     self.idx += 1;
                     if k_idx < 2 {
                         if let Some(km) = self.killers[k_idx] {
                             // Must be legal, quiet, and not TT move
                             if km != self.tt_move.unwrap_or_default() {
                                // Safe check for killer move destination being empty
                                // Access bitboard properly via iteration or mask
                                // chess crate BitBoard is opaque mostly, but targets is a BitBoard
                                let targets = !self.board.color_combined(!self.board.side_to_move());
                                // We want to check if dest is in targets (which are empty/quiet squares)
                                // BitBoard doesn't have get_unchecked or get usually publicly exposed nicely
                                // But we can convert square to BitBoard and AND
                                let dest_bb = chess::BitBoard::from_square(km.get_dest());
                                if (targets & dest_bb) == dest_bb {
                                    // It is a quiet move (dest is empty)
                                     if chess::MoveGen::new_legal(&self.board).any(|m| m == km)
                                     {
                                         return Some(km);
                                     }
                                }
                             }
                         }
                         continue; // Check next killer
                     }
                     self.stage = Stage::Quiets;
                     self.idx = 0;
                },
                Stage::Quiets => {
                    if self.quiets_buffer.is_empty() && self.idx == 0 {
                        self.generate_quiets();
                    }
                    if self.idx < self.quiets_buffer.len() {
                        let mv = self.quiets_buffer[self.idx];
                        self.idx += 1;

                        // Check if already yielded as killer
                        if Some(mv) == self.killers[0] || Some(mv) == self.killers[1] {
                            continue;
                        }

                        return Some(mv);
                    }
                    self.stage = Stage::Done;
                },
                Stage::BadQuiets => {
                    self.stage = Stage::Done;
                },
                Stage::Done => return None,
            }
        }
    }
}
