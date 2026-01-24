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
             // We need to re-generate EP moves specifically or just rely on MoveGen handling it?
             // MoveGen handles EP if the target square is set correctly.
             // But set_iterator_mask takes a BitBoard.
             // Standard chess crate MoveGen with mask *targets excludes EP if dest is empty.
             // We already fixed this in search.rs, need to apply same logic here.

            let dest_idx = if self.board.side_to_move() == chess::Color::White {
                ep_sq.to_index() + 8
            } else {
                ep_sq.to_index() - 8
            };
            let dest = unsafe { chess::Square::new(dest_idx as u8) };

            // Re-create generator with fixed mask
            let mut mask = *targets;
            mask ^= chess::BitBoard::from_square(dest);
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
                    if self.captures_buffer.is_empty() && self.idx == 0 {
                        self.generate_captures();
                    }

                    if self.idx < self.captures_buffer.len() {
                        let mv = self.captures_buffer[self.idx];
                        // Check if SEE >= 0
                        if eval::see(&self.board, mv) >= 0 {
                            self.idx += 1;
                            return Some(mv);
                        } else {
                            // Skip losing captures for now, save for later stage?
                            // For simplicity, we just iterate them in StagedMoveGen but mark them.
                            // Actually, let's just emit them in CapturesLosing
                             self.idx += 1;
                             continue; // Wait, we missed returning it!
                             // We need to store it or split the buffer.
                             // Let's split buffer on generation? No.
                             // Let's just iterate all captures here but ordered.
                             // "Winning" captures are first due to sort.
                             // But wait, sort order puts high SEE first.
                             // So eventually we hit negative SEE.
                        }
                    }

                    // Reset idx for next stage (if we had separate buffers)
                    // Since we have one buffer sorted by score, we just need to iterate it all?
                    // Refinement: Staged generation usually splits "Good Captures" and "Bad Captures"
                    // because we want to search Killers *before* Bad Captures.

                    // RESTART logic for correct staging:
                    // We need to filter captures buffer into Good/Bad
                    // Or iterate it and skip bad, then revisit bad.

                    // Let's redo generation:
                    self.captures_buffer.clear(); // Clear initial empty
                    self.generate_captures(); // Now full

                    // Partition?
                    // Let's stick to simple iterator: Yield all good captures now.
                    // Store bad captures for later?

                    // Simplification for this iteration:
                    // Just yield all captures. Optimization comes from yielding TT move first.
                    // To do strictly TT -> Cap -> Killer -> Quiet, we need that order.

                    // Let's implement full logic:
                    // 1. TT Move (Done)
                    // 2. Captures (Good)
                    // 3. Killers (Non-capture)
                    // 4. Quiets (Good History)
                    // 5. Captures (Bad) - usually skipped in qsearch but needed in main search

                    // Current simplification:
                    // 2. All Captures (Sorted)
                    // 3. Killers
                    // 4. Quiets

                    self.idx = 0;
                    self.stage = Stage::CapturesLosing; // Reuse stage names loosely
                },
                Stage::CapturesLosing => {
                     // Iterate populated captures buffer
                     if self.idx < self.captures_buffer.len() {
                         let mv = self.captures_buffer[self.idx];
                         self.idx += 1;
                         return Some(mv);
                     }
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
