
#[cfg(test)]
mod tests {
    use chess::{Board, MoveGen, ChessMove, Piece, Square, Color, BoardStatus};
    use std::str::FromStr;

    #[test]
    fn test_en_passant_generation_with_mask() {
        // Fen with en passant available:
        // White pawns on e5, Black pawn moves d7-d5. White can capture exd6 e.p.
        // FEN: rnbqkbnr/ppppp3/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1 (wait, f6? let's construct carefully)

        // Let's use Board::from_fen
        // Position: White pawn on e5. Black plays d7-d5.
        // FEN after d7-d5: rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1
        // d6 is the en passant target square.

        let fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1";
        let board = Board::from_str(fen).expect("Valid FEN");

        // Verify en passant square
        println!("En passant square: {:?}", board.en_passant());
        // assert_eq!(board.en_passant(), Some(Square::D6));

        // Verify move is legal
        let ep_move = ChessMove::new(Square::E5, Square::D6, None);
        assert!(MoveGen::new_legal(&board).any(|m| m == ep_move), "En passant move should be legal");

        // Now test the logic from search.rs (with FIX)
        let mut targets = *board.color_combined(!board.side_to_move()); // Enemy pieces only

        // FIX: Add en passant target square
        if let Some(ep_sq) = board.en_passant() {
            // board.en_passant() returns the victim square (where the pawn is).
            // We need the destination square (empty).
            let dest_idx = if board.side_to_move() == Color::White {
                ep_sq.to_index() + 8
            } else {
                ep_sq.to_index() - 8
            };

            // Safety: EP squares are always valid
            let dest = unsafe { Square::new(dest_idx as u8) };

            // Add to mask
            // Assuming BitBoard implements BitOr or can be manipulated
            // BitBoard is usually a wrapper around u64.
            // Let's assume we can create a BitBoard from a square and OR it.
            // Note: BitBoard usually implements BitOr<&BitBoard> or BitOr<BitBoard>.

            // Check if chess::BitBoard has set methods or if we need to combine.
            // targets is a BitBoard (copied).
            // Let's try combining.

            // targets |= BitBoard::from_square(dest); // This might work if impl exists
            // Or targets = targets | BitBoard::from_square(dest);
             targets ^= chess::BitBoard::from_square(dest);
        }

        let mut gen = MoveGen::new_legal(&board);
        gen.set_iterator_mask(targets);
        let captures: Vec<ChessMove> = gen.collect();

        // Check if ep_move is in captures
        let found = captures.contains(&ep_move);

        if !found {
            println!("En Passant move MISSING from filtered generation!");
        } else {
            println!("En Passant move FOUND in filtered generation!");
        }

        assert!(found, "En Passant capture was filtered out by set_iterator_mask!");
    }
}
