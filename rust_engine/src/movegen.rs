use chess::{Board, ChessMove, MoveGen};
use crate::eval;
use crate::search::{KILLERS, HISTORY};
use std::mem::MaybeUninit;

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

#[derive(Clone, Copy, Debug)]
pub struct ScoredMove {
    pub mv: ChessMove,
    pub score: i32,
}

// Fixed-size move list to avoid heap allocation
const MAX_MOVES: usize = 252; // Enough for almost any position

pub struct MoveList {
    moves: [MaybeUninit<ScoredMove>; MAX_MOVES],
    count: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            moves: [MaybeUninit::uninit(); MAX_MOVES],
            count: 0,
        }
    }

    pub fn push(&mut self, mv: ChessMove, score: i32) {
        if self.count < MAX_MOVES {
            self.moves[self.count] = MaybeUninit::new(ScoredMove { mv, score });
            self.count += 1;
        }
    }

    // Returns a mutable slice of the initialized moves
    pub fn as_slice_mut(&mut self) -> &mut [ScoredMove] {
        // Safety: We only access up to self.count, which have been initialized
        unsafe {
            std::mem::transmute(&mut self.moves[..self.count])
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

pub struct StagedMoveGen<'a> {
    board: &'a Board,
    tt_move: Option<ChessMove>,
    killers: [Option<ChessMove>; 2],
    stage: Stage,
    captures: MoveList,
    quiets: MoveList,
    idx: usize,
}

impl<'a> StagedMoveGen<'a> {
    pub fn new(board: &'a Board, tt_move: Option<ChessMove>, ply: u8) -> Self {
        // Fetch killers for this ply
        let killers = KILLERS.with(|k: &std::cell::RefCell<[[Option<ChessMove>; 2]; 64]>|
            k.borrow()[(ply as usize).min(63)]
        );

        Self {
            board,
            tt_move,
            killers,
            stage: Stage::TTMove,
            captures: MoveList::new(),
            quiets: MoveList::new(),
            idx: 0,
        }
    }

    // Generate captures and populate list
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

             let see = eval::see(self.board, m);
             // Higher is better
             let score = eval::mvv_lva_score(self.board, m) + see * 10;
             self.captures.push(m, score);
        }
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

            let from = m.get_source().to_index();
            let to = m.get_dest().to_index();
            let score = HISTORY.with(|h| h.borrow()[from][to]);
            self.quiets.push(m, score);
        }
    }

    // Pick best move from list starting at self.idx
    
  (&mut self, list_type: Stage) -> Option<ChessMove> {
        let list = match list_type {
            Stage::CapturesWinning => self.captures.as_slice_mut(),
            Stage::Quiets => self.quiets.as_slice_mut(),
            _ => return None,
        };

        if self.idx >= list.len() {
            return None;
        }

        // Selection sort: Find best move in remaining portion [idx..len]
        let mut best_idx = self.idx;
        let mut best_score = list[best_idx].score;

        for i in (self.idx + 1)..list.len() {
            if list[i].score > best_score {
                best_score = list[i].score;
                best_idx = i;
            }
        }

        // Swap best to front (idx)
        list.swap(self.idx, best_idx);
        let mv = list[self.idx].mv;
        self.idx += 1;
        Some(mv)
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
                    if self.captures.is_empty() && self.idx == 0 {
                        self.generate_captures();
                    }

                    if let Some(mv) = self.pick_best(Stage::CapturesWinning) {
                         return Some(mv);
                    }

                    // Done with captures
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
                    if self.quiets.is_empty() && self.idx == 0 {
                        self.generate_quiets();
                    }

                    loop {
                         if let Some(mv) = self.pick_best(Stage::Quiets) {
                              // Check if already yielded as killer
                              if Some(mv) == self.killers[0] || Some(mv) == self.killers[1] {
                                  continue;
                              }
                              return Some(mv);
                         }
                         break;
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
