# **The Calculus of Caissa: A Comprehensive Architectural Analysis of the Classical Stockfish Engine (Versions 1–11)**

## **1\. Introduction: The Classical Paradigm in Computational Chess**

The history of computer chess is demarcated by two distinct epochs: the classical era, dominated by symbolic artificial intelligence and alpha-beta search, and the neural era, defined by the advent of Efficiently Updatable Neural Networks (NNUE). While the latter has garnered significant attention since 2020, the former represents nearly seventy years of algorithmic refinement, culminating in the non-NNUE iterations of Stockfish (Versions 1 through 11). These versions constitute the zenith of "handcrafted" chess engines—systems where every heuristic, evaluation term, and pruning parameter was explicitly defined by human programmers and refined through rigorous statistical testing.

Stockfish’s dominance in this era was not the result of a single breakthrough but the aggregation of thousands of incremental improvements to the Minimax algorithm and its evaluation function. Unlike its commercial predecessors, such as Rybka or Fritz, which operated as closed-box systems, Stockfish leveraged an open-source development model that allowed for the distributed verification of logic. This report provides an exhaustive technical analysis of the design principles that propelled Stockfish to the summit of the classical era. It dissects the engine's approach to board representation, the non-linear complexities of its Principal Variation Search (PVS), the theoretical shift from Young Brothers Wait Concept (YBWC) to Lazy Symmetric Multiprocessing (Lazy SMP), and the sophisticated, mathematically tuned parameters of its handcrafted evaluation (HCE).

The evolution from Stockfish 1.0, a fork of Tord Romstad’s Glaurung engine, to Stockfish 11, the final "pure" classical engine, offers a masterclass in software engineering optimization and game theory.1 By treating chess as a problem of resource allocation—specifically, the allocation of processor cycles to the most promising nodes of a game tree that exceeds the number of atoms in the observable universe—the Stockfish developers constructed a machine that could out-calculate any human and eventually every other machine on the planet.

## **2\. Fundamental Data Structures and Board Representation**

The speed of a chess engine is fundamentally capped by the efficiency of its board representation. Every search operation—generating moves, making moves, evaluating positions—relies on querying the state of the board. Stockfish adopted and refined the bitboard representation, a standard initially popularized by the Soviet chess program Kaissa and later refined in engines like Chess 4.0, but optimized it for modern 64-bit processor architectures.

### **2.1 Bitboards and Little-Endian Rank-File Mapping**

Stockfish utilizes bitboards, where the board state is represented by twelve 64-bit integers (one for each piece type and color) and additional integers for occupancy. This representation allows the engine to perform set-wise operations using bitwise logic (AND, OR, XOR, NOT) rather than iterating through arrays. For instance, determining all safe squares for a Knight can be computed in a handful of CPU cycles by masking the Knight's position with pre-computed attack tables.3

The engine employs a Little-Endian Rank-File (LERF) mapping, where square 0 corresponds to a1 and square 63 to h8. This mapping aligns with the register structure of x86-64 processors, allowing for efficient serialization of bitboards (finding the index of the "1" bits) using hardware instructions like popcount (population count) and bit scan forward/reverse (BSF/BSR).4 The efficiency of these low-level operations is paramount; in the classical era, Stockfish could search upwards of 100 million nodes per second (NPS) on standard hardware, a throughput achievable only because the underlying board logic was reduced to bitwise arithmetic.

### **2.2 Magic Bitboards and Sliding Piece Generation**

A critical bottleneck in bitboard engines is generating moves for "sliding" pieces (Rooks, Bishops, Queens), as their attacks depend on the configuration of blocking pieces. Stockfish addresses this using **Magic Bitboards**. This technique utilizes a perfect hashing algorithm to map the occupancy of a rank, file, or diagonal to an attack bitboard.

For a rook on a specific square, the relevant occupancy bits (the other pieces on the same rank and file) are masked and multiplied by a specific 64-bit "magic number." The result, shifted right by a pre-calculated amount, produces a unique index for a lookup table containing the pre-computed attack set. This transforms the sliding piece move generation from a loop-based operation (scanning square by square until a blocker is hit) into a constant-time $O(1)$ memory lookup. This optimization is crucial for the evaluation function, which heavily utilizes mobility and attack maps to determine piece activity.3

### **2.3 State Management and Copy-Make**

Stockfish handles state updates (making and unmaking moves) using a "Copy-Make" approach where necessary, but relies heavily on incremental updates for the evaluation function. The StateInfo structure tracks non-bitboard state data, such as castling rights, en passant squares, the fifty-move rule counter, and the Zobrist hash key. The Zobrist key is a 64-bit signature of the position, incrementally updated by XORing the random keys associated with the moving piece and the squares involved. This key serves as the index for the Transposition Table, allowing the search to recognize identical positions reached via different move orders.6

## **3\. The Search Core: Principal Variation Search and Iterative Deepening**

Stockfish’s search algorithm is a highly optimized variant of Alpha-Beta pruning known as Principal Variation Search (PVS), or NegaScout. The core philosophy of PVS is that if a move ordering heuristic is effective, the first move searched (the Principal Variation or PV move) will likely be the best. Therefore, the search window can be maximally tightened for all subsequent moves to prove they are inferior.

### **3.1 PVS Mechanics and Null Windows**

In a standard Alpha-Beta search, every node is searched with a window $(\\alpha, \\beta)$. PVS optimizes this by searching the first move with the full window $(\\alpha, \\beta)$. For all remaining moves, the engine attempts to prove that they are *not* better than the PV move by searching them with a "null window" or "zero-width window" $(\\alpha, \\alpha+1)$.

If a move searched with $(\\alpha, \\alpha+1)$ fails low (returns a score $\\le \\alpha$), the assumption holds: the move is inferior to the PV move. This constitutes the vast majority of cases in a well-ordered tree. However, if the move fails high (returns a score $\> \\alpha$), the assumption was incorrect; the new move is potentially superior. The engine must then re-search this specific move with the full window $(\\alpha, \\beta)$ to ascertain its true value.8 This "research" penalty is the cost of PVS, but the statistical prevalence of the first move being the best makes the algorithm significantly more efficient than standard Alpha-Beta.

### **3.2 Iterative Deepening and Time Management**

Stockfish does not simply launch a recursive search to a target depth $D$. Instead, it employs **Iterative Deepening**, performing complete searches to depth 1, then depth 2, up to depth $D$. While this appears redundant, the exponential growth of the search tree means that the cost of iterations $1 \\dots D-1$ is negligible compared to iteration $D$ (typically totaling roughly 30-50% of the time of the final iteration).9

Iterative Deepening provides two indispensable advantages:

1. **Anytime Algorithm:** The search can be aborted at any millisecond (e.g., due to time control constraints) and still return the best move found in the deepest *completed* iteration.  
2. **Move Ordering Seeding:** The best move found at depth $D-1$ is fed into the move sorter for depth $D$. This ensures that the PVS assumption (first move is best) is statistically likely to hold, maintaining the efficiency of the null-window searches.9

### **3.3 Aspiration Windows**

To further exploit the stability of Iterative Deepening, Stockfish uses **Aspiration Windows**. Rather than initializing the root search with infinite bounds $(-\\infty, \+\\infty)$, the engine guesses that the score at depth $D$ will be roughly similar to the score at depth $D-1$. The search is initialized with a narrow window $(v \- \\delta, v \+ \\delta)$, where $v$ is the previous score and $\\delta$ is a small margin (e.g., roughly 0.25 pawns).10

If the true score lies within this window, the search concludes rapidly, as many branches are pruned by the tight bounds. If the score falls outside (a "fail-low" or "fail-high"), the engine must widen the window and search again. Stockfish’s implementation is dynamic: if a fail-high occurs, the window might expand unsymmetrically to capture the new evaluation. This technique reduces the effective branching factor by ensuring the alpha-beta cutoff mechanism triggers as early as possible.12

## **4\. Parallelization: The Paradigm Shift to Lazy SMP**

One of the defining architectural evolutions in Stockfish’s history was the transition from the Young Brothers Wait Concept (YBWC) to Lazy Symmetric Multiprocessing (Lazy SMP) with the release of Stockfish 7 in 2016\. This shift represented a departure from deterministic parallelism toward a probabilistic, shared-state model.13

### **4.1 The Limitations of YBWC**

Prior to version 7, Stockfish utilized YBWC, a "work-stealing" algorithm. In YBWC, the search tree is explicitly divided. A "master" thread searches the PV nodes. When it encounters a node with multiple legal moves (the "Young Brothers"), it delegates these moves to idle "slave" threads. The master must then wait for the slaves to complete or for a beta-cutoff to signal that the remaining work is unnecessary.14

While YBWC minimizes redundant search (nodes are rarely searched twice), it imposes heavy synchronization overhead. Threads must constantly acquire locks to check for available work, report results, and update the tree status. As CPU core counts increased, this overhead became a bottleneck, preventing linear scaling. The complexity of managing thread states also made the codebase difficult to maintain and optimize.15

### **4.2 The Mechanics of Lazy SMP**

Lazy SMP abandons the idea of explicit work distribution. Instead, it launches $N$ helper threads that all conceptually search the *same* root position. The threads are "lazy" because they do not coordinate their actions explicitly. The synchronization mechanism is entirely implicit, mediated through the **Shared Transposition Table (TT)**.7

1. **Shared Memory:** The massive Transposition Table (often gigabytes in size) is accessible to all threads.  
2. **Implicit Communication:** When Thread A evaluates a position and stores the result in the TT, Thread B can later retrieve this result. If Thread B encounters the same position (via transposition), it sees the table entry. If the entry's depth is sufficient, Thread B can cut off its search immediately, benefiting from Thread A's work without ever "talking" to Thread A.  
3. **Divergence:** To prevent all threads from performing identical searches, Stockfish introduces deliberate divergence. Threads are initialized with different random seeds (affecting Zobrist hashing and probabilistic decisions) and slightly varied search parameters (e.g., different reduction depths or move ordering biases). This causes the threads to naturally explore different branches of the tree.13

### **4.3 Thread Voting and Scaling**

In Stockfish’s Lazy SMP implementation, the final move decision is not solely the prerogative of the main thread. A **Thread Voting** mechanism is employed where the "best move" is determined by aggregating the results of all threads. The main thread weighs the moves proposed by helper threads based on the depth they achieved and their calculated scores. This makes the engine more robust against "search blindness," where a single thread might miss a tactic due to aggressive pruning; with multiple threads stumbling through the tree, the likelihood of *one* thread finding the refutation increases.13

Tests demonstrated that while Lazy SMP performs significant redundant work (searches identical nodes), the near-zero synchronization overhead allows for much higher Nodes Per Second (NPS). On systems with high core counts, the raw throughput advantage of Lazy SMP overwhelms the efficiency advantage of YBWC.

## **5\. Selectivity: Pruning and Reductions**

The primary strength of Stockfish lies not just in how fast it searches, but in what it chooses *not* to search. The engine employs aggressive selectivity techniques to reduce the effective branching factor from \~35 (the average number of legal moves) to less than 2 in many positions.

### **5.1 Null Move Pruning (NMP)**

Null Move Pruning is based on the "Null Move Observation": in chess, having the right to move is almost always an advantage. If a player passes their turn (makes a "null move") and the resulting position is still statistically winning (score $\\ge \\beta$), then the initial position is likely so strong that a detailed search is unnecessary.20

Stockfish implements adaptive NMP. It reduces the search depth by $R$ plies (typically $R=3$ or more, scaling with depth) after a null move. If this reduced search fails high, the node is pruned. To prevent errors in Zugzwang positions (where moving is disadvantageous, and passing would be better), NMP is disabled in endgames where Zugzwang is likely (e.g., King and Pawn endings).3

### **5.2 Late Move Reductions (LMR)**

LMR is the engine's primary method for handling the width of the game tree. It operates on the assumption that the move ordering heuristics are generally accurate. Therefore, moves appearing late in the sorted list are likely inferior and do not warrant a full-depth search.

Stockfish applies a reduction formula to these moves:

$$\\text{Reduction} \= \\text{Base} \+ \\frac{\\ln(\\text{depth}) \\times \\ln(\\text{move\\\_index})}{\\text{constant}}$$

In Stockfish 11, this formula was tuned via SPSA to roughly:

$$R \= 0.78 \+ \\frac{\\ln(D) \\times \\ln(M)}{2.46}$$

If a move is the 10th move in the list and the depth is 20, it might be searched at depth 14 instead of 19\. If this reduced search yields a score above alpha (a surprise fail-high), the move is re-searched at full depth. This technique allows the engine to scan deep into the game tree along the principal variation while spending minimal time on implausible sidelines.21

### **5.3 Singular Extensions (SE)**

While LMR curtails the search, **Singular Extensions** expand it. This heuristic detects "forced" moves—situations where one move is significantly better than all alternatives. If the Transposition Table suggests a move is good, Stockfish performs a reduced-depth search on all *other* legal moves. If all these alternatives fail low by a significant margin (the "Singular Margin"), the TT move is deemed "singular."

To ensure accuracy in these critical lines, Stockfish *extends* the search depth for the singular move by one ply. This allows the engine to see deeper into forced tactical sequences (mates, heavy material exchanges) without wasting resources on the entire tree. The logic follows the principle that when there is no choice, calculation must be precise.22

### **5.4 ProbCut and Razoring**

**ProbCut (Probabilistic Cutoff)** utilizes the correlation between shallow and deep searches. If a search at reduced depth $D'$ returns a score significantly higher than beta (specifically $\\beta \+ T$, where $T$ is a threshold), the engine infers that a full depth search would also fail high and prunes the node immediately. This is particularly effective for winning tactical shots.8

**Razoring** addresses "frontier nodes" (nodes near the leaf of the search tree). If the static evaluation of a node is significantly below alpha, the engine assumes that no quiet move can improve the position enough to beat alpha. It skips the remaining quiet moves and drops directly into Quiescence Search (Q-Search), which only examines captures and checks. This prevents wasting time on hopeless positions.25

## **6\. Move Ordering Heuristics**

The efficacy of Alpha-Beta pruning is entirely dependent on move ordering. If the best move is searched first, 90% of the tree can be pruned. Stockfish employs a multi-stage **MovePicker** to generate and sort moves dynamically.

### **6.1 The Ordering Hierarchy**

1. **Hash Move:** The best move from the Transposition Table is always searched first. This move has already been vetted by a previous iteration or a transposing search, giving it the highest probability of being best.  
2. **Captures (MVV-LVA):** Captures are ordered by "Most Valuable Victim \- Least Valuable Aggressor." Capturing a Queen with a Pawn is examined before capturing a Pawn with a Queen. Stockfish also employs **SEE (Static Exchange Evaluation)** to prune "bad captures" (captures that lose material immediately) from the main search, delegating them to Q-Search.27  
3. **Killer Moves:** Stockfish maintains "Killer Moves" slots for every ply. These are quiet moves that caused a beta-cutoff at the same search depth in sibling nodes. The assumption is that a move that refutes one line at depth $D$ is a good candidate to refute a sibling line.29  
4. **History Heuristics:** For quiet moves that are not Killers, Stockfish uses a **History Table** (often called a Butterfly Board, 64x64 array). When a quiet move fails high, its score in the table is incremented; when it fails low, it is decremented. Moves are sorted based on their accumulated history score, prioritizing moves that have historically performed well in the current search tree.29

### **6.2 Counter Moves History**

Introduced in Stockfish 7, Counter Moves History adds a second order of logic to the history heuristic. Standard history is index-independent (stateless regarding the previous move). Counter Moves History is indexed by the previous move. The engine asks: "When the opponent played Move X, what was the most effective reply in the past?"  
This allows Stockfish to "learn" specific tactical responses. If e7e5 is consistently a good reply to d2d4, the Counter Moves table will prioritize e7e5 whenever d2d4 is played, regardless of the general history score of e7e5.29

### **6.3 Follow-Up History**

Stockfish 11 further refined this with **Follow-Up History**, which considers the move *two* plies back. This allows the engine to recognize successful plans or maneuvers initiated by its own side, effectively identifying "what move usually works well after I played X and the opponent played Y?".29

## **7\. The Handcrafted Evaluation Function (HCE)**

Before the neural network revolution, Stockfish's "intelligence" was encapsulated in its evaluation function. This was not a simple material counter but a massive, statically typed polynomial function, evaluate(), containing hundreds of terms.

### **7.1 Tapered Evaluation**

Stockfish acknowledges that the value of pieces and patterns changes as the game progresses. A Blocked Pawn is a liability in the endgame but a shield in the middlegame. To model this, Stockfish computes two scores for every feature: a Middle Game (MG) score and an End Game (EG) score.  
The final evaluation is a linear interpolation:

$$\\text{Eval} \= \\frac{\\text{MG} \\times \\text{Phase} \+ \\text{EG} \\times (128 \- \\text{Phase})}{128}$$

The Phase variable starts at 128 (full board) and decreases as material is exchanged. This creates a smooth transition, preventing "horizon effects" where the engine might erroneously sacrifice material solely to cross a binary threshold into an endgame definition.32

### **7.2 Material and Quadratic Imbalances**

Standard evaluation assigns static points (e.g., Pawn=100, Knight=300). Stockfish uses a much higher internal resolution (Pawn \~ 198 midgame) and employs Imbalance Tables.  
Recognizing that piece values are relative, Stockfish calculates Quadratic Imbalance terms. The value of a piece is adjusted based on the presence of every other piece on the board.

$$V\_{piece} \= Base \+ \\sum (C\_{own}\[P\_i\] \\times W\_{same}) \+ \\sum (C\_{opp}\[P\_j\] \\times W\_{opp})$$

This allows the engine to mathematically encode concepts like "The Bishop pair is worth more than two Bishops individually" or "Rooks increase in value when the opponent has fewer minor pieces to block them".34

### **7.3 Piece-Square Tables (PST) and Mobility**

Stockfish uses granular PSTs to define positional play. A Knight on the rim is penalized; a Knight in the center is rewarded. These tables are also tapered.  
Mobility is calculated by counting the safe squares a piece can move to. Crucially, Stockfish excludes squares attacked by enemy pawns from the mobility count, preventing the engine from inflating the value of pieces that have "pseudo-mobility" into dangerous areas. The mobility score is non-linear (e.g., the difference between 0 and 1 move is weighted more heavily than 10 vs 11 moves).36

### **7.4 Pawn Structure Analysis**

The evaluation of pawn structure is one of the most computationally expensive aspects of the HCE. To mitigate this, Stockfish uses a dedicated **Pawn Hash Table**.

* **Connected Bonus:** Stockfish rewards "Connected Pawns" (pawns on adjacent files). The bonus increases with the rank of the pawns (rewarding advanced phalanxes) and is higher if the pawns are opposed (blocking the enemy).38  
* **Levers:** The engine explicitly detects "levers"—a pawn configuration where a capture can open lines (e.g., white pawn on e4, black pawn on d5). Levers are valued highly in the middlegame as they represent dynamic potential.39  
* **Weaknesses:** Backward, Isolated, and Doubled pawns are penalized. However, the penalties are context-sensitive; a doubled pawn is penalized less if it controls critical central squares.40

### **7.5 King Safety: Storms and Shelters**

Stockfish 11’s King Safety is a complex heuristic that scans the "King Zone" (the squares around the King).

1. **Attacker Weighting:** Every enemy piece attacking the zone adds to an "Attacker Sum."  
2. **Safety Table:** The sum is used to look up a penalty in a non-linear table. A single check might be worth 0.5 pawns, but a coordinated attack by Queen and Knight might be worth 4.0 pawns (decisive).  
3. **Pawn Storm:** The engine calculates a "Storm" metric based on the proximity of enemy pawns to the King.  
4. **Pawn Shelter:** It evaluates the friendly pawns in front of the King. "Holes" (squares with no friendly pawn) and "fianchetto gaps" are heavily penalized.36

## **8\. Verification: The Fishtest Framework and SPSA**

The "secret weapon" of Stockfish was not a specific algorithm, but its testing infrastructure. **Fishtest**, launched in 2013, allowed developers to outsource testing to a distributed cloud of volunteers.

### **8.1 SPRT and GSPRT**

Testing chess engines is statistically difficult due to the high draw rate. Fishtest utilizes the **Sequential Probability Ratio Test (SPRT)**.

* **Hypothesis Testing:** A patch is treated as a hypothesis $H\_1$ (better than master) vs $H\_0$ (equal or worse).  
* **Log-Likelihood Ratio:** The system updates a likelihood ratio after every game.  
* Termination: If the ratio exceeds a specific upper bound, the patch is accepted. If it falls below a lower bound, it is rejected.  
  This allows the framework to reject bad patches quickly (often in \<1000 games) while investing more resources to prove smaller Elo gains. Later versions adopted GSPRT (Generalized SPRT), which uses a pentanomial model (Win/Loss/Draw/Win-on-Time/Loss-on-Time) to extract more statistical signal from every game.43

### **8.2 SPSA Tuning**

To tune the thousands of evaluation parameters (e.g., "Bonus for Rook on open file \= 45"), Stockfish uses **Simultaneous Perturbation Stochastic Approximation (SPSA)**.

* **Gradient Descent:** SPSA is a gradient descent optimization method.  
* **Perturbation:** Instead of tweaking one variable at a time (which would take centuries), SPSA perturbs *all* variables simultaneously by a random amount $\\pm \\Delta$. It plays games with the perturbed weights and uses the result to estimate the gradient of the entire parameter vector.  
* **Convergence:** Over hundreds of thousands of games, the parameters converge toward a local optimum. This automated tuning allowed Stockfish 11 to achieve a level of calibration precision that human intuition could never match.44

## **9\. Conclusion**

The architecture of the classical Stockfish engine (Versions 1–11) stands as a monument to the power of heuristic search. By decomposing the game of chess into a hierarchy of bitwise operations, probabilistic search windows, and mathematically tuned evaluation terms, the developers created a system that transcended human capability.

The transition to Lazy SMP solved the hardware scaling problem, proving that shared-state probabilistic parallelism was superior to deterministic work-splitting in large search spaces. The refinement of PVS with Singular Extensions and LMR solved the depth problem, allowing the engine to spot 20-move tactical sequences by ignoring millions of irrelevant branches. Finally, the Handcrafted Evaluation, honed by SPSA and Fishtest, codified the essence of positional chess into a set of linear equations.

While Stockfish 12 eventually introduced the NNUE architecture, replacing the HCE with a neural network, the search and parallelization strategies developed during the classical era remain the engine's backbone. The legacy of Stockfish 1–11 is the proof that with sufficient optimization, a "brute force" system can be refined into an instrument of profound strategic insight.

### ---

**Table 1: Key Algorithmic Features of Stockfish (v1–v11)**

| Component | Feature | Function | Strategic Impact |
| :---- | :---- | :---- | :---- |
| **Search** | **PVS (NegaScout)** | Main search logic | Optimizes search by assuming the first move is best; minimal windows for others. |
| **Search** | **Iterative Deepening** | Search control | Enables anytime best-move retrieval; seeds move ordering for deeper searches. |
| **Search** | **Aspiration Windows** | Pruning | Restricts search bounds around expected score; allows massive pruning if score is stable. |
| **Parallelism** | **Lazy SMP** | Multithreading | Replaced YBWC (v7). Threads share TT but search independently; high scaling on many cores. |
| **Selectivity** | **Singular Extensions** | Extension | Extends search depth for "forced" moves to verify uniqueness and avoid horizon effects. |
| **Selectivity** | **Null Move Pruning** | Pruning | Prunes branches where "passing" is still winning; risks Zugzwang errors (handled by logic). |
| **Ordering** | **Counter Moves** | Heuristic | (v7+) Prioritizes moves that historically refuted the opponent's specific previous move. |
| **Evaluation** | **Tapered Eval** | Scoring | Interpolates between Midgame and Endgame weights based on material phase. |
| **Evaluation** | **Quadratic Imbalance** | Scoring | Evaluates piece value dynamically based on the interaction with other pieces on board. |
| **Tuning** | **SPSA / Fishtest** | Optimization | Automated parameter tuning via distributed cloud testing (SPRT/GSPRT). |

### **Table 2: Comparison of Parallel Search Strategies**

| Strategy | Mechanism | Pros | Cons |
| :---- | :---- | :---- | :---- |
| **YBWC** (Pre-SF7) | **Work Stealing:** Master splits moves at node, slaves help search. Explicit sync. | Zero redundant search; theoretically optimal node count. | High synchronization overhead; poor scaling on \>8 cores; complex code. |
| **Lazy SMP** (Post-SF7) | **Shared State:** All threads search root. Sync via Transposition Table. | Near-zero overhead; linear scaling on high cores; robust against "blind spots." | High redundant search (threads duplicate work); non-deterministic node counts. |

#### **Works cited**

1. Stockfish (chess) \- Wikipedia, accessed January 10, 2026, [https://en.wikipedia.org/wiki/Stockfish\_(chess)](https://en.wikipedia.org/wiki/Stockfish_\(chess\))  
2. I am the first author of Stockfish. Ask me anything. : r/chess \- Reddit, accessed January 10, 2026, [https://www.reddit.com/r/chess/comments/soffd5/i\_am\_the\_first\_author\_of\_stockfish\_ask\_me\_anything/](https://www.reddit.com/r/chess/comments/soffd5/i_am_the_first_author_of_stockfish_ask_me_anything/)  
3. Slow Chess \- Programming Details \- 3DKingdoms, accessed January 10, 2026, [https://3dkingdoms.com/chess/implementation.htm](https://3dkingdoms.com/chess/implementation.htm)  
4. How to compile Stockfish from source code \- GitHub Pages, accessed January 10, 2026, [https://official-stockfish.github.io/docs/stockfish-wiki/Compiling-from-source.html](https://official-stockfish.github.io/docs/stockfish-wiki/Compiling-from-source.html)  
5. official-stockfish/Stockfish: A free and strong UCI chess engine \- GitHub, accessed January 10, 2026, [https://github.com/official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish)  
6. Stockfish/src/search.cpp at master \- GitHub, accessed January 10, 2026, [https://github.com/mcostalba/Stockfish/blob/master/src/search.cpp](https://github.com/mcostalba/Stockfish/blob/master/src/search.cpp)  
7. Shared Hash Table \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Shared\_Hash\_Table](https://www.chessprogramming.org/Shared_Hash_Table)  
8. An Update on Game Tree Research Tutorial 3: Alpha-Beta Search and Enhancements, accessed January 10, 2026, [https://webdocs.cs.ualberta.ca/\~mmueller/courses/2014-AAAI-games-tutorial/slides/AAAI-14-Tutorial-Games-3-AlphaBeta.pdf](https://webdocs.cs.ualberta.ca/~mmueller/courses/2014-AAAI-games-tutorial/slides/AAAI-14-Tutorial-Games-3-AlphaBeta.pdf)  
9. Iterative deepening: what should I do with previous results? \- Chess Stack Exchange, accessed January 10, 2026, [https://chess.stackexchange.com/questions/33448/iterative-deepening-what-should-i-do-with-previous-results](https://chess.stackexchange.com/questions/33448/iterative-deepening-what-should-i-do-with-previous-results)  
10. Aspiration window \- Wikipedia, accessed January 10, 2026, [https://en.wikipedia.org/wiki/Aspiration\_window](https://en.wikipedia.org/wiki/Aspiration_window)  
11. Aspiration Windows \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Aspiration\_Windows](https://www.chessprogramming.org/Aspiration_Windows)  
12. Aspiration Windows & Checkmates : r/chessprogramming \- Reddit, accessed January 10, 2026, [https://www.reddit.com/r/chessprogramming/comments/1hxbi84/aspiration\_windows\_checkmates/](https://www.reddit.com/r/chessprogramming/comments/1hxbi84/aspiration_windows_checkmates/)  
13. Lazy SMP \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Lazy\_SMP](https://www.chessprogramming.org/Lazy_SMP)  
14. A Comparative Study of Parallel Search Techniques in a Chess Engine: Young Brothers Wait Concept or Lazy SMP? \- Diva-portal.org, accessed January 10, 2026, [http://www.diva-portal.org/smash/get/diva2:1947313/FULLTEXT02.pdf](http://www.diva-portal.org/smash/get/diva2:1947313/FULLTEXT02.pdf)  
15. Lazy SMP Better than YBWC? \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=58031](https://talkchess.com/viewtopic.php?t=58031)  
16. Empirical results with Lazy SMP, YBWC, DTS \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=56019](https://talkchess.com/viewtopic.php?t=56019)  
17. SMP, first shot at implementation \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=75088](https://talkchess.com/viewtopic.php?t=75088)  
18. Chess Engine Using LazySMP, accessed January 10, 2026, [https://chess.stackexchange.com/questions/35257/chess-engine-using-lazysmp](https://chess.stackexchange.com/questions/35257/chess-engine-using-lazysmp)  
19. What's the best Lazy SMP logic? \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=69507](https://talkchess.com/viewtopic.php?t=69507)  
20. Threads \- Stockfish Docs, accessed January 10, 2026, [https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html](https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html)  
21. Late Move Reductions \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Late\_Move\_Reductions](https://www.chessprogramming.org/Late_Move_Reductions)  
22. Singular Extensions \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Singular\_Extensions](https://www.chessprogramming.org/Singular_Extensions)  
23. Singular extensions \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/forum/viewtopic.php?p=615940](https://talkchess.com/forum/viewtopic.php?p=615940)  
24. Stockfish/src/search.cpp at master \- GitHub, accessed January 10, 2026, [https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)  
25. Razoring \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Razoring](https://www.chessprogramming.org/Razoring)  
26. Can someone ELI5 how the big chess engines work? Why do engines “miss” moves like in the video below? \- Reddit, accessed January 10, 2026, [https://www.reddit.com/r/chess/comments/kl38bj/can\_someone\_eli5\_how\_the\_big\_chess\_engines\_work/](https://www.reddit.com/r/chess/comments/kl38bj/can_someone_eli5_how_the_big_chess_engines_work/)  
27. Move Ordering \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Move\_Ordering](https://www.chessprogramming.org/Move_Ordering)  
28. Move ordering \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=29966](https://talkchess.com/viewtopic.php?t=29966)  
29. History Heuristic \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/History\_Heuristic](https://www.chessprogramming.org/History_Heuristic)  
30. I Made Stockfish Even Stronger \- Daniel Monroe : r/chessprogramming \- Reddit, accessed January 10, 2026, [https://www.reddit.com/r/chessprogramming/comments/1kb917e/i\_made\_stockfish\_even\_stronger\_daniel\_monroe/](https://www.reddit.com/r/chessprogramming/comments/1kb917e/i_made_stockfish_even_stronger_daniel_monroe/)  
31. Problem with counter move heuristic \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=76255](https://talkchess.com/viewtopic.php?t=76255)  
32. Stockfish Evaluation Guide \- GitHub Pages, accessed January 10, 2026, [https://hxim.github.io/Stockfish-Evaluation-Guide/](https://hxim.github.io/Stockfish-Evaluation-Guide/)  
33. Dissecting Stockfish Part 2: In-Depth Look at a chess engine | by Antoine Champion, accessed January 10, 2026, [https://medium.com/data-science/dissecting-stockfish-part-2-in-depth-look-at-a-chess-engine-2643cdc35c9a](https://medium.com/data-science/dissecting-stockfish-part-2-in-depth-look-at-a-chess-engine-2643cdc35c9a)  
34. Stockfish \- material balance/imbalance evaluation \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=34159](https://talkchess.com/viewtopic.php?t=34159)  
35. Value of pieces • page 3/3 • General Chess Discussion \- Lichess.org, accessed January 10, 2026, [https://lichess.org/forum/general-chess-discussion/value-of-pieces?page=3](https://lichess.org/forum/general-chess-discussion/value-of-pieces?page=3)  
36. King Safety \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/King\_Safety](https://www.chessprogramming.org/King_Safety)  
37. How to write a chess evaluation function?, accessed January 10, 2026, [https://chess.stackexchange.com/questions/17957/how-to-write-a-chess-evaluation-function](https://chess.stackexchange.com/questions/17957/how-to-write-a-chess-evaluation-function)  
38. Connected Pawns \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Connected\_Pawns](https://www.chessprogramming.org/Connected_Pawns)  
39. How does Stockfish, a powerful chess engine, evaluate a position by assigning a numerical score to it, representing the advantage or disadvantage of one side over the other? \- Quora, accessed January 10, 2026, [https://www.quora.com/How-does-Stockfish-a-powerful-chess-engine-evaluate-a-position-by-assigning-a-numerical-score-to-it-representing-the-advantage-or-disadvantage-of-one-side-over-the-other](https://www.quora.com/How-does-Stockfish-a-powerful-chess-engine-evaluate-a-position-by-assigning-a-numerical-score-to-it-representing-the-advantage-or-disadvantage-of-one-side-over-the-other)  
40. Pawn Structures in Chess: What Are They and How to Evaluate? \- Chesscul, accessed January 10, 2026, [https://chesscul.com/en/pawn-structure/](https://chesscul.com/en/pawn-structure/)  
41. Pawn Structure \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/Pawn\_Structure](https://www.chessprogramming.org/Pawn_Structure)  
42. Underwhelming results from king safety evaluation \- TalkChess.com, accessed January 10, 2026, [https://talkchess.com/viewtopic.php?t=82407](https://talkchess.com/viewtopic.php?t=82407)  
43. Statistical Methods and Algorithms in Fishtest | Stockfish Docs \- GitHub Pages, accessed January 10, 2026, [https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)  
44. Creating my first test \- Stockfish Docs, accessed January 10, 2026, [https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html)  
45. SPSA \- Chessprogramming wiki, accessed January 10, 2026, [https://www.chessprogramming.org/SPSA](https://www.chessprogramming.org/SPSA)