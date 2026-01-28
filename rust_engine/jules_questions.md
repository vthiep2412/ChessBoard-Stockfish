# Jules Session - Quality Score Improvement

## Current State
- **Quality Score**: 70.7% (24/30 moves) - EXPERT tier
- **Target**: 90%+ GRANDMASTER tier
- **NPS**: 3.5-4.6M (good speed)
- **Build**: Passing ✅

## Recent Changes Applied
1. ✅ Delta pruning deferred for promotions (Qodo suggestion)
2. ✅ SEE pruning removed (was buggy)
3. ✅ Material delta pruning added as SEE replacement
4. ✅ Killer move comparison bug fixed (`unwrap_or_default()` → `map_or`)
5. ✅ MAX_MOVES overflow warning added

## Failing Positions (Need Investigation)
1. **bk_pawn_break**: Engine: `c3g3`, SF: `d4d5` - Missing pawn break
2. **mid_knight_maneuver**: Engine: `d1b3`, SF: `d2e4` - Knight positioning
3. **ara_pawn_storm**: Engine: `g4g5`, SF: `e5f6` - Pawn storm tactics
4. **ara_bishop_sac_mate**: Engine: `e2c2`, SF: `d3h7` - Missing bishop sacrifice
5. **lct_advanced_push**: Engine: `h1h2`, SF: `d5d6` - Passed pawn push

## Potential Root Causes (NOT SURE, YOU MUST AUDIT)
1. **King Safety** weights may be too low (missing sacrifices)
2. **Passed Pawn** evaluation may not incentivize advancement enough
3. **Attack Weights** may need tuning (`ATTACK_WEIGHT` in eval.rs)
4. **Mobility** evaluation might undervalue piece activity

## Files to Review (NOT SURE, YOU MUST AUDIT)
- `src/eval.rs` - Evaluation constants and king safety
- `src/search.rs` - Pruning heuristics (LMP, LMR, NMP)
- `src/pst.rs` - Piece-square tables

## Suggested Actions (NOT SURE, YOU MUST AUDIT)
1. Analyze failing positions to understand what evaluation factor is missing
2. Consider increasing `ATTACK_WEIGHT` (currently `[0, 9, 9, 12, 20]`)
3. Review passed pawn bonus for rank 6+
4. Check if lazy eval cutoffs are too aggressive

## Build & Test
```bash
cd rust_engine
.\build.bat
python benchmark.py 14
```
