# Implementation Plan: Advanced Non-NNUE Optimizations

## Overview
This plan outlines the roadmap for "Objective 4" to elevate the engine's strength to rival Stockfish 12-14 levels (approx. 3300+ ELO) using classical techniques. These optimizations focus on search efficiency and evaluation precision without relying on neural networks.

## 1. Search Optimizations (Estimated ELO Gain: ~100-150)

### A. Advanced Late Move Reductions (LMR)
**Complexity:** Medium
**Benefit:** 50-80 ELO
**Implementation:**
*   **History-Based Reduction:** Reduce moves *less* if they have a high History Heuristic score (they are historically good).
*   **Tactical Awareness:** Do not reduce moves that give check or capture loose pieces (SEE > 0).
*   **Formula Tuning:** Implement `R = K + log(depth) * log(move_count) / C`. Tune constants `K` and `C` via SPSA (Simultaneous Perturbation Stochastic Approximation).

### B. Singular Extensions (SE)
**Complexity:** High
**Benefit:** 30-50 ELO
**Implementation:**
*   **Concept:** If the TT move evaluates significantly better than the second-best move (singular), extend the search depth by 1.
*   **Mechanism:** Run a reduced-depth search on alternative moves with a lower bound (beta = TT_score - margin). If they all fail low, the move is singular.
*   **Effect:** Prevents pruning tactical variations where only one move holds the position.

### C. ProbCut
**Complexity:** Medium
**Benefit:** 10-20 ELO
**Implementation:**
*   **Concept:** Uses a shallow search (e.g., depth 4) to predict if a deep search (e.g., depth 8) will cause a beta cutoff.
*   **Formula:** `v = qsearch(alpha - T * sigma, beta + T * sigma)`. If `v >= beta + T * sigma`, prune.

## 2. Evaluation Optimizations (Estimated ELO Gain: ~80-120)

### A. Pawn Structure Hashing
**Complexity:** Medium
**Benefit:** Huge Speedup (20-30% NPS increase)
**Implementation:**
*   **Concept:** Pawn structure changes rarely. Store expensive pawn eval terms (passed pawns, backward pawns, chains) in a specialized Hash Table keyed by `pawn_key`.
*   **Integration:** In `evaluate()`, check the Pawn Hash Table first. If hit, reuse scores.

### B. Sophisticated King Safety
**Complexity:** High
**Benefit:** 40-60 ELO
**Implementation:**
*   **Attack Units:** Define "attack units" for each piece type (e.g., Knight=2, Rook=3, Queen=5).
*   **Zone of Control:** Calculate sum of attacks on the King's ring (squares adjacent to King).
*   **Pattern Weighting:**
    *   Safe checks (checks that can be delivered safely).
    *   Weak squares (holes) around the king.
    *   Pawn storms (enemy pawns advancing on the king's file).
*   **Formula:** `SafetyScore = w1 * AttackUnits^2 / 100`. (Quadratic scaling is crucial).

### C. Mobility & Trapped Pieces
**Complexity:** Medium
**Benefit:** 20-30 ELO
**Implementation:**
*   **Mobility:** Count safe squares available to pieces (excluding squares attacked by enemy pawns).
*   **Trapped Pieces:** Penalize Bishops/Rooks blocked by own pawns or trapped inside enemy territory (e.g., Knight trapped on a8).

## 3. Move Ordering (Estimated ELO Gain: ~30-50)

### A. Counter-Move History
**Complexity:** Low
**Benefit:** 10-20 ELO
**Implementation:**
*   Track which move was good in response to the *previous* move.
*   `CounterMove[prev_move] = good_response_move`.
*   Give a bonus to this move in sorting.

### B. Capture History
**Complexity:** Low
**Benefit:** 10-15 ELO
**Implementation:**
*   Similar to History Heuristic but for captures. Prioritize captures that have historically caused cutoffs, even if MVV-LVA/SEE suggests otherwise.

## Implementation Schedule

1.  **Phase 1 (Search):** Singular Extensions & Advanced LMR. (Highest ELO/effort ratio).
2.  **Phase 2 (Speed):** Pawn Structure Hashing. (Enables more complex eval terms).
3.  **Phase 3 (Eval):** Advanced King Safety & Mobility.
4.  **Phase 4 (Tuning):** Run an automated tuner (SPSA) on all evaluation weights.

## Approval Request
Please approve this plan to begin implementation of **Phase 1 (Search Optimizations)**.
