import os
import urllib.request
import zipfile
import shutil

# URL for Stockfish 16.1 AVX2 (Windows)
# Using direct GitHub link if possible, or fallback to a known reliable mirror if GH fails logic
# Let's try Sourceforge or GitHub direct for 16.1
URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-windows-x86-64-avx2.zip"
DEST_ZIP = "stockfish.zip"
EXTRACT_DIR = "stockfish_bin"

def download_file():
    print(f"Downloading {URL}...")
    try:
        with urllib.request.urlopen(URL) as response, open(DEST_ZIP, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")
        # Fallback to SourceForge link if GH fails? Usually GH is fine.
        return False
    return True

def extract_file():
    print(f"Extracting to {EXTRACT_DIR}...")
    try:
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)
        with zipfile.ZipFile(DEST_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
        
        # Find the exe
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for file in files:
                if file.endswith(".exe") and "stockfish" in file.lower():
                    print(f"Found Stockfish binary: {os.path.join(root, file)}")
                    # Move to top level for convenience
                    shutil.move(os.path.join(root, file), "stockfish.exe")
                    print("Moved to ./stockfish.exe")
                    return
        print("Warning: No .exe found in zip!")
        
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    if download_file():
        extract_file()
        if os.path.exists(DEST_ZIP):
            os.remove(DEST_ZIP)
