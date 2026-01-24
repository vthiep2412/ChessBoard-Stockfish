import chess
import chess.engine
import random

def generate_tuning_data(output_file="tuning_data.epd", num_positions=10000):
    """
    Generates random positions by playing random games.
    Saves them in EPD format with the result.
    """
    with open(output_file, "w") as f:
        games_played = 0
        while games_played < num_positions / 50: # Avg 50 moves per game
            board = chess.Board()
            game_moves = []

            while not board.is_game_over():
                try:
                    move = random.choice(list(board.legal_moves))
                    board.push(move)
                    game_moves.append(board.fen())
                except IndexError:
                    break

            result = board.result()
            # Convert result to 1.0, 0.5, 0.0
            if result == "1-0":
                val = "1.0"
            elif result == "0-1":
                val = "0.0"
            else:
                val = "0.5"

            # Save positions from the game (skip first 8 moves usually to avoid opening book bias)
            for fen in game_moves[8:]:
                f.write(f"{fen} c9 \"{val}\";\n")

            games_played += 1
            if games_played % 10 == 0:
                print(f"Generated {games_played} games...")

if __name__ == "__main__":
    print("Generating tuning data (random self-play)...")
    # Generating a small set for demonstration; usually needs 1M+
    generate_tuning_data(num_positions=100)
    print("Done. Saved to tuning_data.epd")
