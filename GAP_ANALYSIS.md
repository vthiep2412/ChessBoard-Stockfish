# Gap Analysis: Rust Engine vs. Stockfish

**Targets:**
*   **Evaluation:** Stockfish 11 (Hand-Crafted Evaluation Peak)
*   **Speed/MoveGen:** Stockfish 17 (Modern Search/MoveGen Architecture)

## 1. Evaluation Audit (vs. Stockfish 11)

Stockfish 11's evaluation is highly dense, featuring complex interaction between terms. The current `rust_engine` evaluation is functionally "stubbed" by comparison, lacking critical terms and depth in existing ones.

### Top 3 Missing/Stubbed Terms

#### A. Threats & Hanging Pieces (MISSING)
*   **SF11 Implementation:** Dedicated `threats()` function.
    *   **Logic:** Calculates bonuses for:
        *   Minor pieces attacking major pieces (e.g., Knight attacking Rook).
        *   "Hanging" pieces (weak pieces not strongly protected and under attack).
        *   Safe pawn threats against enemy pieces.
        *   Restricted enemy piece movement.
*   **My Engine:** **Completely Missing.** There is no logic to reward attacking valuable pieces or punish leaving pieces en prise (hanging). This leads to tactical blindness in quiet positions.

#### B. King Safety (STUBBED)
*   **SF11 Implementation:** Extremely complex `king()` function.
    *   **Logic:**
        *   **Safe Checks:** Distinguishes between checks that are "safe" (attacker not easily captured) vs "unsafe".
        *   **King Ring:** Uses a bitboard of squares around the king + squares in front, removing those defended by friendly pawns.
        *   **King Flank:** Analyzes attacks and defenses on the files adjacent to the king.
        *   **King Danger:** Accumulates "danger units" based on attackers' weight and count, mapped to a non-linear penalty curve.
*   **My Engine:** Primitive stub.
    *   **Current Logic:** Only checks if enemy pieces attack the immediate 8 squares around the king. It ignores whether the checks are safe, ignores the wider "flank", and lacks the non-linear "danger" accumulation that makes SF11 paranoid about attacks.

#### C. Passed Pawns (STUBBED)
*   **SF11 Implementation:** Sophisticated `passed()` function.
    *   **Logic:**
        *   **King Proximity:** Adjusts bonus based on how close the friendly/enemy kings are to the pawn.
        *   **Path Safety:** Checks if the path to promotion is controlled by enemy pieces.
        *   **Blocker Support:** Increases bonus if the square in front of the pawn is supported.
        *   **Candidate Passers:** Penalizes pawns that aren't fully passed yet but have potential.
*   **My Engine:** Simple bonus.
    *   **Current Logic:** Binary check `(front_span & enemy_pawns == 0)`. If true, adds a fixed rank-based bonus. It fails to account for whether the pawn can *actually* advance or if the king can catch it.

---

## 2. Speed/MoveGen Audit (vs. Stockfish 17)

**Bottleneck Confirmation:** Confirmed. The `rust_engine` suffers from significant overhead due to heap allocation and inefficient sorting strategies in the hot loop.

### Architecture Comparison

| Feature | Stockfish 17 (`movepick.h` / `movegen.h`) | My Engine (`movegen.rs`) | Impact |
| :--- | :--- | :--- | :--- |
| **Memory Storage** | **Fixed-size Array (Stack)**<br>`ExtMove moves[MAX_MOVES]` | **Heap Vector**<br>`Vec<ChessMove>` | **Critical.** `Vec::with_capacity` and `push` cause allocation/reallocation overhead at *every* node (millions/sec). |
| **Sorting Strategy** | **Partial / Lazy**<br>`partial_insertion_sort` (sorts only top N moves) or `select<Best>` (lazy swap). | **Full Sort**<br>`vec.sort_by_key(...)` sorts the *entire* list every time. | **High.** Sorting bad moves (that will be pruned anyway) is wasted CPU cycles. |
| **Generation** | **Pointer Filling**<br>`generate(pos, ExtMove* list)` fills a pre-allocated buffer. | **Vector Push**<br>`gen.push(move)` involves bounds checking and pointer indirection. | **Moderate.** Pointer arithmetic is faster than vector management. |

### Recommendation for `movegen.rs`

Refactor `StagedMoveGen` to match SF17's "MovePicker" architecture:

1.  **Remove `Vec`:** Replace `captures_buffer` and `quiets_buffer` with a fixed-size array on the struct:
    ```rust
    const MAX_MOVES: usize = 256;
    struct MoveList {
        moves: [ScoredMove; MAX_MOVES],
        count: usize,
    }
    ```
2.  **Pointer-based Generation:** Pass a mutable slice `&mut [ScoredMove]` to generation functions instead of returning a Vec.
3.  **Lazy/Partial Sorting:**
    *   For Captures: Implement a `partial_insertion_sort` or simple selection sort that only sorts the top few captures if the list is long, or just use a faster unstable sort on the slice.
    *   For Quiets: Do not sort the entire list immediately. Pick the best move one by one (Selection Sort) or sort only the first N moves.

---

## Summary of Action Plan (Pending Approval)

1.  **Prioritize Speed (MoveGen):** The 5000ms vs 0ms gap is structural. Fixing this requires rewriting `movegen.rs` to remove `Vec` usage.
2.  **Prioritize Brains (Eval):** Once speed is acceptable, implement **Threats** and upgrade **King Safety** to match SF11's logic density.
