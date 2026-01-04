import os
import urllib.request
import time
from PIL import Image

# Chess.com 'Neo' theme pieces (Clean, PNG)
# Base URL: https://images.chesscomfiles.com/chess-themes/pieces/neo/150/{piece}.png
# piece: wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk

MAPPING = {
    'wP': 'wp', 'wN': 'wn', 'wB': 'wb', 'wR': 'wr', 'wQ': 'wq', 'wK': 'wk',
    'bP': 'bp', 'bN': 'bn', 'bB': 'bb', 'bR': 'br', 'bQ': 'bq', 'bK': 'bk',
}

DEST_DIR = "assets/pieces"

def download_assets():
    if not os.path.exists(DEST_DIR):
        print(f"Creating {DEST_DIR}...")
        os.makedirs(DEST_DIR, exist_ok=True)

    headers = {'User-Agent': 'Mozilla/5.0'}

    for filename_key, url_key in MAPPING.items():
        url = f"https://images.chesscomfiles.com/chess-themes/pieces/neo/150/{url_key}.png"
        filename = os.path.join(DEST_DIR, f"{filename_key}.png")
        
        print(f"Downloading {filename_key} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = response.read()
                with open(filename, 'wb') as f:
                    f.write(data)
            
            # Verify Image
            try:
                img = Image.open(filename)
                img.verify()
                print(f"  -> Valid Image: {img.format} {img.size}")
            except Exception as e:
                print(f"  -> INVALID IMAGE: {e}")
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to download {filename_key}: {e}")

if __name__ == "__main__":
    download_assets()
