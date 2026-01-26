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

// Stack-based Move Buffer to avoid allocations
struct MoveBuffer {
    moves: [Option<ChessMove>; 256],
    count: usize,
}

impl MoveBuffer {
    fn new() -> Self {
        Self {
            moves: [None; 256],
            count: 0,
        }
    }

    fn push(&mut self, mv: ChessMove) {
        if self.count < 256 {
            self.moves[self.count] = Some(mv);
            self.count += 1;
        }
    }

    fn get(&self, idx: usize) -> Option<ChessMove> {
        if idx < self.count {
            self.moves[idx]
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn len(&self) -> usize {
        self.count
    }

    fn sort_by_score<F>(&mut self, score_fn: F)
    where F: Fn(ChessMove) -> i32
    {
        // Simple insertion sort or standard sort on slice
        // We can access the valid slice
        let slice = &mut self.moves[0..self.count];
        slice.sort_by_cached_key(|m| -score_fn(m.unwrap()));
    }
}

pub struct StagedMoveGen<'a> {
    board: &'a Board,
    tt_move: Option<ChessMove>,
    killers: [Option<ChessMove>; 2],
    stage: Stage,
    captures_buffer: MoveBuffer,
    quiets_buffer: MoveBuffer,
    idx: usize,
}

impl<'a> StagedMoveGen<'a> {
    pub fn new(board: &'a Board, tt_move: Option<ChessMove>, ply: u8) -> Self {
        // Fetch killers for this ply
        // Explicitly type closure argument to satisfy compiler
        let killers = KILLERS.with(|k: &std::cell::RefCell<[[Option<ChessMove>; 2]; 64]>|
            k.borrow()[(ply as usize).min(63)]
        );

        Self {
            board,
            tt_move,
            killers,
            stage: Stage::TTMove,
            captures_buffer: MoveBuffer::new(),
            quiets_buffer: MoveBuffer::new(),
            idx: 0,
        }
    }

    // Generate captures and populate buffer
    fn generate_captures(&mut self) {
        let targets = self.board.color_combined(!self.board.side_to_move());
        let mut gen = MoveGen::new_legal(self.board);
        gen.set_iterator_mask(*targets);

        // Add EP capture if available
        if let Some(ep_sq) = self.board.en_passant() {
            let mask = *targets | chess::BitBoard::from_square(ep_sq);
            gen.set_iterator_mask(mask);
        }

        // Collect
        for m in gen {
             if Some(m) == self.tt_move { continue; }
             self.captures_buffer.push(m);
        }

        // Sort by MVV-LVA + SEE
        let board = self.board;
        self.captures_buffer.sort_by_score(|m| {
            let see = eval::see(board, m);
            // Higher is better
            eval::mvv_lva_score(board, m) + see * 10
        });
    }

    // Generate quiets
    fn generate_quiets(&mut self) {
        let targets = !self.board.color_combined(!self.board.side_to_move()); // Non-captures
        let mut gen = MoveGen::new_legal(self.board);
        gen.set_iterator_mask(targets); // Only quiets

        for m in gen {
            if Some(m) == self.tt_move { continue; }

            // Skip ONLY EP captures (pawn moving diagonally to EP square)
            if Some(m.get_dest()) == self.board.en_passant()
                && self.board.piece_on(m.get_source()) == Some(chess::Piece::Pawn)
                && m.get_source().get_file() != m.get_dest().get_file()
            {
                continue;
            }

            self.quiets_buffer.push(m);
        }

        // Sort by History
        // Use thread-local access
        self.quiets_buffer.sort_by_score(|m| {
             let from = m.get_source().to_index();
             let to = m.get_dest().to_index();
             HISTORY.with(|h: &std::cell::RefCell<[[i32; 64]; 64]>|
                h.borrow()[from][to]
             )
        });
    }
}

impl<'a> Iterator for StagedMoveGen<'a> {
    type Item = ChessMove;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::CapturesWinning;
                    if let Some(mv) = self.tt_move {
                        // Verify legality (TT move might be from collision)
                        if self.board.legal(mv) {
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
                        let mv = self.captures_buffer.get(self.idx).unwrap();
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
                             if km != self.tt_move.unwrap_or_default()
                                // Check for quiet move: destination must be empty
                                && self.board.piece_on(km.get_dest()).is_none()
                                // Verify legality
                                && self.board.legal(km)
                             {
                                 return Some(km);
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
                        let mv = self.quiets_buffer.get(self.idx).unwrap();
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
