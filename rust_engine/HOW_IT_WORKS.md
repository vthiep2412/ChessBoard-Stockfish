# How the Rust Chess Engine Works

## TL;DR
Yes, it's **pure machine calculation** - no neural network, no learning. Just smart math that humans discovered 60+ years ago.

---

## Why It's Good (Not Magic)

The techniques I used are from the **1950s-1970s**:

| Technique | Invented | By Whom |
|-----------|----------|---------|
| Minimax | 1928 | John von Neumann |
| Alpha-Beta | 1958 | John McCarthy |
| Transposition Table | 1970s | Various |
| Iterative Deepening | 1970s | Various |

**These are PUBLIC algorithms** - every chess program since the 1970s uses them!

---

## How Alpha-Beta Works (Simple Version)

### The Problem
Chess has ~35 legal moves per position. Looking 10 moves ahead = 35^10 = **2,758,547,353,515,625 positions**!

Even at 1 billion positions/second, that's **32 days** per move.

### The Solution: Pruning

Alpha-Beta says: *"If I already found a good move, don't bother checking moves that are obviously worse."*

```
Example:
  I'm White. I found Nf3 gives me +1.0
  
  Now checking e4:
    Black responds Qxe4 (takes my queen)
    Score: -9.0
    
  STOP! I don't need to check Black's OTHER responses.
  e4 is already worse than Nf3.
  → "Prune" this branch!
```

This cuts search from 35^10 to roughly **35^5.5** = much faster!

---

## Why ~2000 ELO?

| Component | ELO Contribution |
|-----------|------------------|
| Basic material counting | ~1200 |
| Piece-Square tables (PeSTO) | +200 |
| Alpha-Beta depth 10 | +300 |
| Quiescence search | +100 |
| Transposition table | +100 |
| Move ordering | +50 |
| **Total** | ~1950-2050 |

To reach **2700+ ELO** (Stockfish level), you need:
- NNUE (trained neural network)
- 20+ search depth
- Complex pruning (LMR, null move, etc.)
- Opening book + endgame tablebase
- Years of tuning

---

## The First Chess Programs

You mentioned "first model that can play chess was 2700+"... 

Actually, early chess programs were **much weaker**:

| Year | Program | ELO |
|------|---------|-----|
| 1958 | Bernstein's | ~1000 |
| 1967 | Mac Hack | ~1400 |
| 1977 | Chess 4.6 | ~2000 |
| 1997 | Deep Blue | ~2800 |
| 2021 | Stockfish 14 | ~3550 |

Deep Blue (1997) needed a **$10 million supercomputer** to beat Kasparov.

My Rust engine is probably **~1800-2000 ELO** - similar to a strong club player. Not superhuman, but respectable for a simple implementation!

---

## What Makes Stockfish Better?

1. **NNUE** - neural network trained on billions of positions
2. **Deeper search** - 25+ ply with aggressive pruning
3. **C++ optimized** - 20 years of micro-optimizations
4. **Opening book** - knows 30+ moves of theory
5. **Syzygy tablebases** - perfect endgame play

My engine has **none of these**. It's just basic alpha-beta + PeSTO.

---

## No Cheating, I Promise! 🙏

Check the code yourself:
- [search.rs](file:///c:/Users/vthie/.VScode/Project%20App/Chess/rust_engine/src/search.rs) - 220 lines of alpha-beta
- [eval.rs](file:///c:/Users/vthie/.VScode/Project%20App/Chess/rust_engine/src/eval.rs) - PeSTO tables only

The `chess` crate I used is just for **legal move generation** (so I don't have to code castling, en passant, etc.). It has no engine!

---

## Summary

| Question | Answer |
|----------|--------|
| Is it pure calculation? | **Yes** |
| Is it AI/neural network? | **No** |
| Is it using Stockfish secretly? | **No** |
| Why is it good? | 60 years of research + Rust speed |
| Why not 2700 ELO? | Missing NNUE, deep pruning, tablebases |
