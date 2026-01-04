import requests
import os
from tqdm import tqdm

# List of top players (Lichess usernames)
# We use Lichess because their API is open, rate-limit friendly, and allows bulk PGN downloads easily.
# Chess.com has an API bit it's often stricter or requires more complex pagination for bulk data.
TOP_PLAYERS = [
    "DrNykterstein",  # Magnus Carlsen
    "Alireza2003",    # Alireza Firouzja
    "Hikaru",         # Hikaru Nakamura (Note: sometimes accounts are closed/renamed)
    "RebeccaHarris",  # Daniel Naroditsky
    "Penguingim1",    # Andrew Tang
    "LyonBeast",      # Maxime Vachier-Lagrave
    "lachesisQ",      # Ian Nepomniachtchi
    "FabianoCaruana", # Fabiano Caruana
    "nihalsarin",     # Nihal Sarin
    "Night-King96"    # Oleksandr Bortnyk
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "grandmaster_games.pgn")

def download_games():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Downloading games for {len(TOP_PLAYERS)} top players...")
    
    # Clear existing file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    total_games = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for player in TOP_PLAYERS:
            print(f"Fetching games for {player}...")
            # Limit to 2000 games per player for better training data
            # Filter: Rapid and Classical only (exclude Blitz/Bullet to minimize blunders)
            url = f"https://lichess.org/api/games/user/{player}?max=2000&perfType=rapid,classical&pgnInJson=true"
            
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    out.write(response.text)
                    out.write("\n\n")
                    total_games += 2000 # Approx
                else:
                    print(f"Failed to fetch {player}: {response.status_code}")
            except Exception as e:
                print(f"Error fetching {player}: {e}")
                
    print(f"\nDownload complete! Saved to {OUTPUT_FILE}")
    print(f"Approx {total_games} games available for training.")

if __name__ == "__main__":
    download_games()
