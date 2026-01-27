import rust_engine
import sys

POSITIONS = [
    ("Root", "r1bq1rk1/pppp1ppp/5n2/2b5/2B1P3/5Q2/PPPP1PPP/RNB1K2R w KQ - 6 6"),
    ("Castle (g1)", "r1bq1rk1/pppp1ppp/5n2/2b5/2B1P3/5Q2/PPPP1PPP/RNB2RK1 b - - 7 6"),
    ("Run (f1)", "r1bq1rk1/pppp1ppp/5n2/2b5/2B1P3/5Q2/PPPP1PPP/RNB2K1R b - - 7 6"),
    ("Develop (c3)", "r1bq1rk1/pppp1ppp/5n2/2b5/2B1P3/2N2Q2/PPPP1PPP/R1B1K2R b KQ - 7 6")
]

rust_engine.set_debug(True)

for name, fen in POSITIONS:
    print(f"\n--- Evaluating {name} ---")
    try:
        score = rust_engine.evaluate(fen)
        print(f"Total Score: {score}")
    except Exception as e:
        print(f"Error: {e}")
