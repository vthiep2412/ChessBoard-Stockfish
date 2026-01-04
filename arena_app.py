import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import chess
import chess.pgn
import subprocess
import sys
import threading
import os
import time
from PIL import Image, ImageTk, ImageDraw

# ==============================================================================
# CONFIG & ASSETS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, "assets/pieces")
ENGINES_DIR = os.path.join(SCRIPT_DIR, "engines")

PIECE_NAMES = {
    'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
    'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'
}

# ==============================================================================
# UCI ENGINE INTERFACE (Reused)
# ==============================================================================

class UCIEngine:
    def __init__(self, path, name="Engine"):
        self.path = path
        self.name = name
        self.process = None
        self._reader_thread = None
        self._output_queue = None
        self._running = False
        
        try:
            import queue
            self._output_queue = queue.Queue()
            
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            if path.endswith('.bat'):
                py_path = path[:-4] + '.py'
                if os.path.exists(py_path):
                    cmd = [sys.executable, py_path]
                else:
                    cmd = path
            elif path.endswith('.py'):
                cmd = [sys.executable, path]
            else:
                cmd = path
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                startupinfo=startupinfo
            )
            
            self._running = True
            self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self._reader_thread.start()
            
            self._send("uci")
            self._wait_for("uciok", timeout=10)
            self._send("isready")
            # Wait longer for first load (PyTorch Lazy Init)
            self._wait_for("readyok", timeout=60)
        except Exception as e:
            print(f"Failed to start engine {name} at {path}: {e}")
            self._running = False
            self.process = None
    
    def _read_output(self):
        try:
            while self._running and self.process:
                line = self.process.stdout.readline()
                if line:
                    self._output_queue.put(line.strip())
                elif self.process.poll() is not None:
                    break
        except:
            pass
    
    def _get_line(self, timeout=60):
        import queue
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _send(self, cmd):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
            except OSError:
                pass
    
    def _wait_for(self, keyword, timeout=60):
        start = time.time()
        while self.process and time.time() - start < timeout:
            remaining = timeout - (time.time() - start)
            if remaining <= 0: break
            line = self._get_line(timeout=min(remaining, 1))
            if line and keyword in line: return line
        return None
    
    def set_option(self, name, value):
        self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok", timeout=10)

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=5)
    
    def get_best_move(self, fen, depth=15, timeout=120):
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        
        start = time.time()
        while self.process and time.time() - start < timeout:
            remaining = timeout - (time.time() - start)
            line = self._get_line(timeout=min(remaining, 1))
            if line and line.startswith("bestmove"):
                parts = line.split()
                best = parts[1] if len(parts) > 1 else None
                return best
        
        self._send("stop")
        line = self._get_line(timeout=2)
        if line and line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else None
        return None
    
    def quit(self):
        self._running = False
        if self.process:
            self._send("quit")
            try: 
                self.process.wait(timeout=2)
            except: 
                self.process.terminate()
            self.process = None

# ==============================================================================
# ARENA APP
# ==============================================================================

class ChessBoardFrame(tk.Frame):
    SQUARE_SIZE = 50 # Smaller for dual view
    BOARD_SIZE = SQUARE_SIZE * 8
    
    LIGHT_COLOR = "#E8EDF9"
    DARK_COLOR = "#B7C0D8"
    LAST_MOVE_COLOR = "#bacb44"

    def __init__(self, parent, title, pieces_images):
        super().__init__(parent, bg="#272522")
        self.pieces_images = pieces_images
        self.board = chess.Board()
        
        # UI
        tk.Label(self, text=title, font=("Arial", 12, "bold"), bg="#272522", fg="white").pack(pady=5)
        
        self.canvas = tk.Canvas(self, width=self.BOARD_SIZE, height=self.BOARD_SIZE, bg="#312E2B", highlightthickness=0)
        self.canvas.pack()
        
        self.status_label = tk.Label(self, text="Ready", font=("Consolas", 10), bg="#272522", fg="#888")
        self.status_label.pack(pady=5)
        
        self.render()

    def update_board(self, board, status_text=""):
        self.board = board
        self.status_label.config(text=status_text)
        self.render()

    def render(self):
        self.canvas.delete("all")
        last_move = self.board.peek() if self.board.move_stack else None
        
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                x = file * self.SQUARE_SIZE
                y = rank * self.SQUARE_SIZE
                is_light = (file + rank) % 2 == 0
                bg = self.LIGHT_COLOR if is_light else self.DARK_COLOR
                
                if last_move and (sq == last_move.from_square or sq == last_move.to_square):
                    bg = self.LAST_MOVE_COLOR
                
                self.canvas.create_rectangle(x, y, x + self.SQUARE_SIZE, y + self.SQUARE_SIZE, fill=bg, outline="")
                
                piece = self.board.piece_at(sq)
                if piece:
                    img = self.pieces_images.get(piece.symbol())
                    if img: self.canvas.create_image(x + self.SQUARE_SIZE//2, y + self.SQUARE_SIZE//2, image=img)

class ArenaApp:
    def __init__(self):
        self.root = ttk.Window(themename="darkly")
        self.root.title("Antigravity Training Arena (Quad Core)")
        self.root.geometry("1100x800")
        
        self.pieces_images = {}
        self._load_assets()
        
        # Engines
        self.engine1_name = "AntigravityZero"
        # Find path
        self.engine1_path = os.path.join(ENGINES_DIR, "AntigravityZero.bat")
        
        self.engine2_name = "Stockfish"
        self.engine2_path = os.path.join(ENGINES_DIR, "stockfish-windows-x86-64-avx2.exe") # Default guess
        # Scan for stockfish if default wrong
        if not os.path.exists(self.engine2_path):
             for f in os.listdir(ENGINES_DIR):
                 if "stockfish" in f.lower() and f.endswith(".exe"):
                     self.engine2_path = os.path.join(ENGINES_DIR, f)
                     break
        
        self.is_running = False
        self.threads = []
        
        self.stats = {
            "AntigravityZero": {"Wins": 0, "Losses": 0, "Draws": 0},
            "Stockfish": {"Wins": 0, "Losses": 0, "Draws": 0}
        }
        self.stats_lock = threading.Lock()
        
        # UI Layout
        controls = tk.Frame(self.root, bg="#272522")
        controls.pack(fill="x", padx=10, pady=10)
        
        self.btn_start = ttk.Button(controls, text="Start Arena (4x)", command=self.toggle_arena, bootstyle="success")
        self.btn_start.pack(side="left", padx=5)
        
        self.score_label = ttk.Label(controls, text="Score: Antigravity 0 - 0 Stockfish (0 Draws)", font=("Arial", 11, "bold"))
        self.score_label.pack(side="left", padx=20)
        
        # Grid View for 4 Boards
        boards_frame = tk.Frame(self.root, bg="#272522")
        boards_frame.pack(fill="both", expand=True)
        
        # Board 1: Antigravity (White)
        self.board1 = ChessBoardFrame(boards_frame, f"Game 1: {self.engine1_name} (W)", self.pieces_images)
        self.board1.grid(row=0, column=0, padx=5, pady=5)
        
        # Board 2: Stockfish (White)
        self.board2 = ChessBoardFrame(boards_frame, f"Game 2: {self.engine2_name} (W)", self.pieces_images)
        self.board2.grid(row=0, column=1, padx=5, pady=5)

        # Board 3: Antigravity (White)
        self.board3 = ChessBoardFrame(boards_frame, f"Game 3: {self.engine1_name} (W)", self.pieces_images)
        self.board3.grid(row=1, column=0, padx=5, pady=5)

        # Board 4: Stockfish (White)
        self.board4 = ChessBoardFrame(boards_frame, f"Game 4: {self.engine2_name} (W)", self.pieces_images)
        self.board4.grid(row=1, column=1, padx=5, pady=5)

    def update_score(self, winner, loser, is_draw=False):
        with self.stats_lock:
            if is_draw:
                self.stats[self.engine1_name]["Draws"] += 1
                self.stats[self.engine2_name]["Draws"] += 1
            else:
                self.stats[winner]["Wins"] += 1
                self.stats[loser]["Losses"] += 1
            
            # Update UI
            w = self.stats[self.engine1_name]["Wins"]
            l = self.stats[self.engine1_name]["Losses"]
            d = self.stats[self.engine1_name]["Draws"]
            self.score_label.config(text=f"Score: Antigravity {w} - {l} Stockfish ({d} Draws)")
            
            # Log to CSV
            try:
                log_file = os.path.join(SCRIPT_DIR, "arena_stats.csv")
                with open(log_file, "a") as f:
                    if os.stat(log_file).st_size == 0:
                        f.write("Time,Winner,Loser,Result\n")
                    
                    res_str = "Draw" if is_draw else f"{winner} beats {loser}"
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{winner if not is_draw else 'Draw'},{loser if not is_draw else 'Draw'},{res_str}\n")
            except Exception as e:
                print(f"Logging failed: {e}")

    def save_pgn(self, board, white_name, black_name, result):
        try:
            pgn_dir = os.path.join(SCRIPT_DIR, "data", "arena_games")
            os.makedirs(pgn_dir, exist_ok=True)
            
            game = chess.pgn.Game()
            game.headers["Event"] = "Antigravity Arena"
            game.headers["White"] = white_name
            game.headers["Black"] = black_name
            game.headers["Result"] = result
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            game.headers["Time"] = time.strftime("%H:%M:%S")
            
            # Reconstruct game from move stack
            node = game
            for move in board.move_stack:
                node = node.add_variation(move)
            
            filename = f"{white_name}_vs_{black_name}_{int(time.time())}.pgn"
            filepath = os.path.join(pgn_dir, filename)
            
            with open(filepath, "w") as f:
                exporter = chess.pgn.FileExporter(f)
                game.accept(exporter)
        except Exception as e:
            print(f"Failed to save PGN: {e}")

    def _load_assets(self):
        SQUARE_SIZE = 50
        if not os.path.exists(ASSET_DIR): os.makedirs(ASSET_DIR, exist_ok=True)
        for symbol, fname in PIECE_NAMES.items():
            path = os.path.join(ASSET_DIR, f"{fname}.png")
            if os.path.exists(path):
                img = Image.open(path)
                if img.mode != 'RGBA': img = img.convert('RGBA')
                img = img.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
                self.pieces_images[symbol] = ImageTk.PhotoImage(img)
            else:
                self.pieces_images[symbol] = self._generate_fallback(symbol, SQUARE_SIZE)

    def _generate_fallback(self, symbol, size):
        img = Image.new("RGBA", (size, size), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        is_white = symbol.isupper()
        color = "white" if is_white else "black"
        outline = "black" if is_white else "white"
        margin = 8
        draw.ellipse([margin, margin, size-margin, size-margin], fill=color, outline=outline, width=2)
        return ImageTk.PhotoImage(img)

    def toggle_arena(self):
        if self.is_running:
            self.is_running = False
            self.btn_start.config(text="Start Arena (4x)", bootstyle="success")
        else:
            self.is_running = True
            self.btn_start.config(text="Stop Arena", bootstyle="danger")
            # Start 4 threads
            # Mix colors: 1 & 3 are Antigravity White. 2 & 4 are Stockfish White.
            t1 = threading.Thread(target=self.run_match, args=(self.board1, self.engine1_path, self.engine2_path, self.engine1_name, self.engine2_name), daemon=True)
            t2 = threading.Thread(target=self.run_match, args=(self.board2, self.engine2_path, self.engine1_path, self.engine2_name, self.engine1_name), daemon=True)
            t3 = threading.Thread(target=self.run_match, args=(self.board3, self.engine1_path, self.engine2_path, self.engine1_name, self.engine2_name), daemon=True)
            t4 = threading.Thread(target=self.run_match, args=(self.board4, self.engine2_path, self.engine1_path, self.engine2_name, self.engine1_name), daemon=True)
            
            self.threads = [t1, t2, t3, t4]
            for t in self.threads:
                t.start()

    def run_match(self, board_ui, white_engine_path, black_engine_path, white_name, black_name):
        """Runs a continuous loop of games between two engines."""
        
        # Init engines
        w_eng = UCIEngine(white_engine_path, "White")
        b_eng = UCIEngine(black_engine_path, "Black")
        
        # Configure
        # Skill level for stockfish (User requested MAX level)
        if "stockfish" in white_engine_path.lower(): w_eng.set_option("Skill Level", 20) 
        if "stockfish" in black_engine_path.lower(): b_eng.set_option("Skill Level", 20)
        
        while self.is_running:
            w_eng.new_game()
            b_eng.new_game()
            board = chess.Board()
            
            board_ui.update_board(board, "Starting...")
            
            while not board.is_game_over() and self.is_running:
                # White Move
                fen = board.fen()
                move_uci = w_eng.get_best_move(fen, depth=10) # Fast games
                if not move_uci: break
                board.push(chess.Move.from_uci(move_uci))
                self.root.after(0, lambda b=board.copy(): board_ui.update_board(b, f"White moved {move_uci}"))
                if board.is_game_over(): break
                time.sleep(0.1) # UI Delay
                
                # Black Move
                fen = board.fen()
                move_uci = b_eng.get_best_move(fen, depth=10)
                if not move_uci: break
                board.push(chess.Move.from_uci(move_uci))
                self.root.after(0, lambda b=board.copy(): board_ui.update_board(b, f"Black moved {move_uci}"))
                time.sleep(0.1)
                
            if self.is_running:
                # Game Over
                res = board.result()
                board_ui.update_board(board, f"Game Over: {res}")
                
                # Update Stats
                winner = None
                loser = None
                is_draw = False
                
                if res == "1-0":
                    winner = white_name
                    loser = black_name
                elif res == "0-1":
                    winner = black_name
                    loser = white_name
                else:
                    is_draw = True
                
                self.root.after(0, lambda: self.update_score(winner, loser, is_draw))
                
                # Save PGN for learning
                self.save_pgn(board, white_name, black_name, res)
                
                # Trigger Auto-Train (Online Learning)
                # Run in separate process to not block UI/Engine, but wait? No, let it run background.
                # Only trigger if Antigravity was playing
                if "Antigravity" in white_name or "Antigravity" in black_name:
                    try:
                        print("Skipping online training to preserve performance.")
                        # subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, "src", "train_online.py")])
                    except Exception as e:
                        print(f"Failed to trigger training: {e}")
                
                time.sleep(3) # Pause before restart
        
            # Allow UI to refresh?
            # We are in thread, so we don't block UI.
            # But we should sleep briefly to not look instant?
            # time.sleep(0.1)
        
        # DO NOT QUIT ENGINES HERE. KEEP ALIVE.
        # w_eng.quit()
        # b_eng.quit()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ArenaApp()
    app.run()
