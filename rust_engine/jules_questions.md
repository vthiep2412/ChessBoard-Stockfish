# Questions for Jules - Chess Engine Tuning

Hi Jules! I'm helping tune a Rust chess engine and we're stuck at ~66% move quality (vs Stockfish reference). The engine is fast (3-5M NPS) but makes suboptimal moves. Could you help diagnose?

## Current Situation

- **Engine**: Rust chess engine with Alpha-Beta + TT + LMR/LMP pruning
- **Benchmark**: 30 test positions, depth 16, compared to Stockfish's top 5 moves
- **Problem**: Quality Score is 66.7% (ADVANCED rating), target is >90% (GRANDMASTER)
- **Speed**: Very fast (3.5M+ NPS), so we have performance budget to trade for accuracy

## Key Files

- `src/search.rs`: Negamax with Alpha-Beta, TT, NMP, LMR, LMP, Singular Extensions
- `src/eval.rs`: Material + PST + Pawn Structure + Mobility + King Safety
- `benchmark.py`: Test harness comparing moves to cached Stockfish responses

## Recent Changes Made

1. **Incremental EvalState**: `evaluate_lazy` now uses passed `EvalState` instead of rebuilding (O(1) vs O(64))
2. **Quiescence Search**: Now uses `is_tactical(capture OR promotion)` instead of just `is_capture`
3. **LMP Relaxed**: Base from 3 → 24 (less aggressive pruning)
4. **LMR Relaxed**: Start at depth 4 (was 3), history divisor 8192 (was 2048)
5. **Removed Lazy Eval from `evaluate_with_state`**: Force full King Safety/Mobility checks

## Specific Questions

1. **Move Ordering**: Is our MVV-LVA + History + Killers + Countermove heuristic sufficient? Should we add a different scoring method?

2. **Quiescence Search**: Is there a risk we're searching too many nodes in Q-search that don't matter? Should we add a Delta Pruning margin?

3. **TT Interaction**: Could stale TT entries from incremental updates be causing issues? We only clear TT at startup, not between positions.

4. **Evaluation Weights**: Do the weights in `eval.rs` (e.g., King Safety, Mobility) seem balanced? Could they be causing the engine to undervalue tactics?

5. **LMR Re-search**: When LMR fails high, we do a full re-search. Is this implementation correct, or could there be a bug causing us to miss good moves?

## Sample Failing Position

```
Position: mid_bishop_development
FEN: r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7
Our Move: d1c2
SF Top 5: d1a4, d1c2, h2h3, e3e4, g1h1
Result: ★2 (Top 2, but not Top 1)
```

We often get Top 2-3 instead of Top 1, suggesting we're close but missing something subtle.

## What Would Help

- Any ideas on parameter tuning (margins, reduction formulas)?
- Suggestions for additional search features (IID, Probcut tuning, etc.)?
- Evaluation term adjustments?

Thanks for any insights!
