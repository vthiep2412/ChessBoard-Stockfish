import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import chess
import chess.pgn
import subprocess
import sys
import threading
import os
import time
import datetime
import math
import json
from PIL import Image, ImageTk, ImageDraw

# ==============================================================================
# CONFIG & ASSETS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, "assets/pieces")
ENGINES_DIR = os.path.join(SCRIPT_DIR, "engines")
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "settings.json")

PIECE_NAMES = {
    'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
    'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'
}

# ==============================================================================
# UCI ENGINE INTERFACE
# ==============================================================================

class UCIEngine:
    def __init__(self, path, name="Engine"):
        self.path = path
        self.name = name
        self.process = None
        self.is_pondering = False
        self.ponder_move = None
        self._reader_thread = None
        self._output_queue = None
        self._running = False
        
        try:
            import queue
            self._output_queue = queue.Queue()
            
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # Handle Python-based engines
            if path.endswith('.bat'):
                # For batch files, check if there's a .py file with same name
                py_path = path[:-4] + '.py'
                if os.path.exists(py_path):
                    # Run Python script directly
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
            
            # Start background reader thread
            self._running = True
            self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self._reader_thread.start()
            
            self._send("uci")
            self._wait_for("uciok", timeout=60)
            self._send("isready")
            self._wait_for("readyok", timeout=180) # Increased to 180s for slow PyTorch load
        except Exception as e:
            print(f"Failed to start engine {name} at {path}: {e}")
            self._running = False
            self.process = None
    
    def _read_output(self):
        """Background thread to read engine output."""
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
        """Get a line from the output queue with timeout."""
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
    
    def _wait_for(self, keyword, timeout=30):
        """Wait for a specific keyword in engine output."""
        start = time.time()
        while self.process and time.time() - start < timeout:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                break
            line = self._get_line(timeout=min(remaining, 1))
            if line and keyword in line:
                return line
        return None
    
    def set_option(self, name, value):
        self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok", timeout=60) # Increased for safety

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=60) # Increased for safety
    
    def get_best_move(self, fen, depth=15, timeout=120):
        """Get best move using depth-based search. Returns (bestmove, ponder_move)."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        
        start = time.time()
        while self.process and time.time() - start < timeout:
            remaining = timeout - (time.time() - start)
            line = self._get_line(timeout=min(remaining, 1))
            if line and line.startswith("bestmove"):
                parts = line.split()
                best = parts[1] if len(parts) > 1 else None
                ponder = None
                if "ponder" in parts:
                    ponder_idx = parts.index("ponder")
                    if ponder_idx + 1 < len(parts):
                        ponder = parts[ponder_idx + 1]
                return best, ponder
        
        # Timeout - send stop and try to get response
        self._send("stop")
        line = self._get_line(timeout=2)
        if line and line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else None, None
        return None, None
    
    def start_pondering(self, fen, ponder_move):
        """Start pondering on expected opponent move."""
        self.is_pondering = True
        self.ponder_move = ponder_move
        self._send(f"position fen {fen} moves {ponder_move}")
        self._send("go ponder")
    
    def stop_pondering(self):
        """Stop pondering and return the result."""
        if self.is_pondering:
            self._send("stop")
            self.is_pondering = False
            # Read bestmove response with timeout
            start = time.time()
            while time.time() - start < 5:
                line = self._get_line(timeout=1)
                if line and line.startswith("bestmove"):
                    parts = line.split()
                    best = parts[1] if len(parts) > 1 else None
                    ponder = None
                    if "ponder" in parts:
                        ponder_idx = parts.index("ponder")
                        if ponder_idx + 1 < len(parts):
                            ponder = parts[ponder_idx + 1]
                    return best, ponder
        return None, None
    
    def ponderhit(self):
        """Signal that the expected ponder move was played."""
        if self.is_pondering:
            self._send("ponderhit")
            self.is_pondering = False
            # Wait for bestmove with timeout
            start = time.time()
            while time.time() - start < 120:  # Allow full search time
                line = self._get_line(timeout=1)
                if line and line.startswith("bestmove"):
                    parts = line.split()
                    best = parts[1] if len(parts) > 1 else None
                    ponder = None
                    if "ponder" in parts:
                        ponder_idx = parts.index("ponder")
                        if ponder_idx + 1 < len(parts):
                            ponder = parts[ponder_idx + 1]
                    return best, ponder
        return None, None
    
    def quit(self):
        self._running = False
        if self.process:
            if self.is_pondering:
                self._send("stop")
            self._send("quit")
            try:
                self.process.wait(timeout=2)
            except:
                self.process.terminate()
            self.process = None

# ==============================================================================
# MAIN CHESS APP
# ==============================================================================

class ChessApp:
    SQUARE_SIZE = 80
    BOARD_SIZE = SQUARE_SIZE * 8
    
    # Colors
    LIGHT_COLOR = "#E8EDF9"
    DARK_COLOR = "#B7C0D8"
    SELECTED_COLOR = "#F7EC45"
    LEGAL_COLOR = "#a9d162" 
    LAST_MOVE_COLOR = "#bacb44"
    BG_COLOR = "#312E2B"
    PANEL_COLOR = "#272522"
    TEXT_COLOR = "#FFFFFF"

    def __init__(self):
        # Use ttkbootstrap with dark theme
        self.root = ttk.Window(themename="darkly")
        self.root.title("Python Chess - Stockfish & Friends")
        self.root.resizable(False, False)
        
        # Game State
        self.board = chess.Board()
        self.game_mode = "PvP" 
        self.player_side = chess.WHITE 
        self.engines = {} 
        self.game_running = False
        
        # Player names (for display)
        self.white_player_name = "Player 1"
        self.black_player_name = "Player 2"
        
        # Settings (saved to file)
        self.ponder_enabled = tk.BooleanVar(value=True)
        self.engine_depth = tk.IntVar(value=15)
        self.engine_contempt = tk.IntVar(value=50)
        self.engine_skill = tk.IntVar(value=20)
        
        # In-game settings (not saved)
        self.show_evaluation = tk.BooleanVar(value=False)
        self.current_eval = 0.0  # Target evaluation value
        self.display_eval = 0.0  # Current displayed value (for animation)
        self.eval_engine = None  # Dedicated engine for evaluation
        self.eval_depth = 18  # Depth for evaluation
        
        # Pondering state
        self.ponder_move = None
        self.is_pondering = False
        
        # CvC move sync event
        self.cvc_move_done = threading.Event()
        
        # Load saved settings
        self._load_settings()
        
        # UI State
        self.pieces_images = {}
        self._load_assets()
        self.selected_square = None
        self.legal_moves = []
        
        # Build Frames
        self.container = tk.Frame(self.root, bg=self.BG_COLOR)
        self.container.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.menu_frame = self._build_menu()
        self.game_frame = self._build_game()
        
        self._show_menu()
    
    def _load_settings(self):
        """Load settings from JSON file."""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                self.ponder_enabled.set(data.get('ponder_enabled', True))
                self.engine_depth.set(data.get('engine_depth', 15))
                self.engine_contempt.set(data.get('engine_contempt', 50))
                self.engine_skill.set(data.get('engine_skill', 20))
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def _save_settings(self):
        """Save settings to JSON file."""
        try:
            data = {
                'ponder_enabled': self.ponder_enabled.get(),
                'engine_depth': self.engine_depth.get(),
                'engine_contempt': self.engine_contempt.get(),
                'engine_skill': self.engine_skill.get()
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def _load_assets(self):
        """Load piece images or generate fallbacks."""
        if not os.path.exists(ASSET_DIR):
            os.makedirs(ASSET_DIR, exist_ok=True)
            
        for symbol, fname in PIECE_NAMES.items():
            path = os.path.join(ASSET_DIR, f"{fname}.png")
            loaded = False
            
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    img.verify() 
                    img = Image.open(path)  # Reopen after verify
                    # Convert to RGBA for proper transparency handling
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img = img.resize((self.SQUARE_SIZE, self.SQUARE_SIZE), Image.Resampling.LANCZOS)
                    self.pieces_images[symbol] = ImageTk.PhotoImage(img)
                    loaded = True
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if not loaded:
                self.pieces_images[symbol] = self._generate_fallback_piece(symbol)

    def _generate_fallback_piece(self, symbol):
        img = Image.new("RGBA", (self.SQUARE_SIZE, self.SQUARE_SIZE), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        is_white = symbol.isupper()
        color = "white" if is_white else "black"
        outline = "black" if is_white else "white"
        margin = 10
        draw.ellipse([margin, margin, self.SQUARE_SIZE-margin, self.SQUARE_SIZE-margin], 
                     fill=color, outline=outline, width=3)
        return ImageTk.PhotoImage(img)

    # --------------------------------------------------------------------------
    # MENU INTERFACE
    # --------------------------------------------------------------------------
    
    def _build_menu(self):
        frame = tk.Frame(self.container, bg=self.BG_COLOR)
        
        tk.Label(frame, text="CHESS", font=("Helvetica", 32, "bold"), 
                 bg=self.BG_COLOR, fg=self.TEXT_COLOR).pack(pady=(20, 40))
        
        modes_frame = tk.LabelFrame(frame, text="Game Mode", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR, padx=20, pady=20)
        modes_frame.pack(fill="x", padx=50)
        
        self.mode_var = tk.StringVar(value="PvC")
        
        tk.Radiobutton(modes_frame, text="Player vs Player", variable=self.mode_var, value="PvP", 
                      bg=self.PANEL_COLOR, fg=self.TEXT_COLOR, selectcolor=self.BG_COLOR, 
                      command=self._update_menu_options).pack(anchor="w", pady=5)
        
        tk.Radiobutton(modes_frame, text="Player vs Computer", variable=self.mode_var, value="PvC", 
                      bg=self.PANEL_COLOR, fg=self.TEXT_COLOR, selectcolor=self.BG_COLOR,
                      command=self._update_menu_options).pack(anchor="w", pady=5)
        
        self.pvc_frame = tk.Frame(modes_frame, bg=self.PANEL_COLOR)
        self.pvc_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(self.pvc_frame, text="Side:", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).grid(row=0, column=0, sticky="w")
        self.pvc_side_var = tk.StringVar(value="White")
        ttk.Combobox(self.pvc_frame, textvariable=self.pvc_side_var, values=["White", "Black"], state="readonly", width=10).grid(row=0, column=1, padx=10)
        
        tk.Label(self.pvc_frame, text="Engine:", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).grid(row=1, column=0, sticky="w", pady=5)
        self.pvc_engine_var = tk.StringVar()
        self.pvc_engine_combo = ttk.Combobox(self.pvc_frame, textvariable=self.pvc_engine_var, state="readonly", width=15)
        self.pvc_engine_combo.grid(row=1, column=1, padx=10)

        tk.Radiobutton(modes_frame, text="Computer vs Computer", variable=self.mode_var, value="CvC", 
                      bg=self.PANEL_COLOR, fg=self.TEXT_COLOR, selectcolor=self.BG_COLOR,
                      command=self._update_menu_options).pack(anchor="w", pady=5)
        
        self.cvc_frame = tk.Frame(modes_frame, bg=self.PANEL_COLOR)
        
        self.available_engines = self._scan_engines()
        if not self.available_engines:
            tk.Label(self.cvc_frame, text="No engines found in engines/ folder!", bg="red", fg="white").pack()
        else:
            tk.Label(self.cvc_frame, text="White Engine:", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).grid(row=0, column=0, sticky="w")
            self.white_engine_var = tk.StringVar(value=self.available_engines[0])
            self.white_combo = ttk.Combobox(self.cvc_frame, textvariable=self.white_engine_var, values=self.available_engines, state="readonly")
            self.white_combo.grid(row=0, column=1, padx=5, pady=2)
            self.white_combo.bind("<<ComboboxSelected>>", self._update_engine_dropdowns)

            tk.Label(self.cvc_frame, text="Black Engine:", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).grid(row=1, column=0, sticky="w")
            self.black_engine_var = tk.StringVar(value="" if len(self.available_engines) < 2 else self.available_engines[1])
            self.black_combo = ttk.Combobox(self.cvc_frame, textvariable=self.black_engine_var, values=self.available_engines, state="readonly")
            self.black_combo.grid(row=1, column=1, padx=5, pady=2)
            self._update_engine_dropdowns()

        # Initialize PvC engine dropdown (since PvC is the default mode)
        self.pvc_engine_combo['values'] = self.available_engines
        if self.available_engines:
            self.pvc_engine_var.set(self.available_engines[0])

        tk.Button(frame, text="PLAY", font=("Arial", 14, "bold"), bg="#81B64C", fg="white", 
                 activebackground="#45753C", activeforeground="white", relief="flat", height=2, width=20,
                 command=self._start_game).pack(pady=30)
        
        # Settings gear button in bottom right (use Label for transparent look)
        gear_label = tk.Label(frame, text="⚙", font=("Arial", 16), bg=self.BG_COLOR, 
                              fg=self.TEXT_COLOR, cursor="hand2")
        gear_label.pack(side="right", anchor="se", padx=10, pady=10)
        gear_label.bind("<Button-1>", lambda e: self._open_settings())
                 
        return frame

    def _scan_engines(self):
        """Scan engines folder and return list of engine names."""
        if not os.path.exists(ENGINES_DIR):
            return []
        engines = []
        for f in os.listdir(ENGINES_DIR):
            if f.endswith(".exe"):
                engines.append(f[:-4])  # Remove .exe extension
            elif f.endswith(".bat"):
                engines.append(f[:-4])  # Remove .bat extension
        return engines

    def _update_menu_options(self):
        mode = self.mode_var.get()
        # Refresh available engines
        self.available_engines = self._scan_engines()
        
        if mode == "PvC":
            self.pvc_frame.pack(fill="x", padx=20, pady=5)
            self.cvc_frame.pack_forget()
            # Update PvC engine dropdown
            self.pvc_engine_combo['values'] = self.available_engines
            if self.available_engines and not self.pvc_engine_var.get():
                self.pvc_engine_var.set(self.available_engines[0])
        elif mode == "CvC":
            self.pvc_frame.pack_forget()
            self.cvc_frame.pack(fill="x", padx=20, pady=5)
            self._update_engine_dropdowns()
        else: # PvP
            self.pvc_frame.pack_forget()
            self.cvc_frame.pack_forget()
    
    def _update_engine_dropdowns(self, event=None):
        if not self.available_engines: return
        white_sel = self.white_engine_var.get()
        black_opts = [e for e in self.available_engines if e != white_sel]
        self.black_combo['values'] = black_opts
        current_black = self.black_engine_var.get()
        if current_black == white_sel or current_black not in black_opts:
            if black_opts: self.black_engine_var.set(black_opts[0])
            else: self.black_engine_var.set("")
    
    def _open_settings(self):
        """Open settings dialog - changes only apply when OK is clicked."""
        dialog = ttk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.geometry("340x380")
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Temporary variables (copy current values)
        temp_ponder = tk.BooleanVar(value=self.ponder_enabled.get())
        temp_depth = tk.IntVar(value=self.engine_depth.get())
        temp_skill = tk.IntVar(value=self.engine_skill.get())
        temp_contempt = tk.IntVar(value=self.engine_contempt.get())
        
        ttk.Label(dialog, text="Engine Settings", font=("Arial", 14, "bold")).pack(pady=(15, 15))
        
        # Ponder setting - Toggle Switch
        ponder_frame = ttk.Frame(dialog)
        ponder_frame.pack(fill="x", padx=20, pady=8)
        ttk.Checkbutton(ponder_frame, text="Enable Pondering", variable=temp_ponder,
                        bootstyle="success-round-toggle").pack(anchor="w")
        ttk.Label(ponder_frame, text="Engine thinks during your turn", 
                  foreground="#888", font=("Arial", 9)).pack(anchor="w", padx=5)
        
        # Depth setting with value label
        depth_frame = ttk.Frame(dialog)
        depth_frame.pack(fill="x", padx=20, pady=8)
        ttk.Label(depth_frame, text="Search Depth:").pack(side="left")
        depth_val_label = ttk.Label(depth_frame, text=str(temp_depth.get()), width=3)
        depth_val_label.pack(side="right")
        depth_scale = ttk.Scale(depth_frame, from_=1, to=30, variable=temp_depth, 
                                bootstyle="success", length=140,
                                command=lambda v: depth_val_label.config(text=str(int(float(v)))))
        depth_scale.pack(side="right", padx=5)
        
        # Skill Level setting with value label
        skill_frame = ttk.Frame(dialog)
        skill_frame.pack(fill="x", padx=20, pady=8)
        ttk.Label(skill_frame, text="Skill Level:").pack(side="left")
        skill_val_label = ttk.Label(skill_frame, text=str(temp_skill.get()), width=3)
        skill_val_label.pack(side="right")
        skill_scale = ttk.Scale(skill_frame, from_=0, to=20, variable=temp_skill,
                                bootstyle="info", length=140,
                                command=lambda v: skill_val_label.config(text=str(int(float(v)))))
        skill_scale.pack(side="right", padx=5)
        ttk.Label(dialog, text="0=weakest, 20=strongest", 
                  foreground="#888", font=("Arial", 9)).pack(anchor="w", padx=25)
        
        # Contempt setting with value label
        contempt_frame = ttk.Frame(dialog)
        contempt_frame.pack(fill="x", padx=20, pady=8)
        ttk.Label(contempt_frame, text="Aggressiveness:").pack(side="left")
        contempt_val_label = ttk.Label(contempt_frame, text=str(temp_contempt.get()), width=3)
        contempt_val_label.pack(side="right")
        contempt_scale = ttk.Scale(contempt_frame, from_=0, to=100, variable=temp_contempt,
                                   bootstyle="danger", length=140,
                                   command=lambda v: contempt_val_label.config(text=str(int(float(v)))))
        contempt_scale.pack(side="right", padx=5)
        ttk.Label(dialog, text="Higher = avoids draws, plays riskier", 
                  foreground="#888", font=("Arial", 9)).pack(anchor="w", padx=25)
        
        # Apply and save on OK
        def apply_and_close():
            self.ponder_enabled.set(temp_ponder.get())
            self.engine_depth.set(int(temp_depth.get()))
            self.engine_skill.set(int(temp_skill.get()))
            self.engine_contempt.set(int(temp_contempt.get()))
            self._save_settings()
            dialog.destroy()
        
        ttk.Button(dialog, text="OK", command=apply_and_close, 
                   bootstyle="success", width=10).pack(pady=15)
        
        # Closing via X discards changes
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

    def _show_menu(self):
        self.game_frame.pack_forget()
        self.menu_frame.pack(fill="both", expand=True)

    # --------------------------------------------------------------------------
    # GAME INTERFACE
    # --------------------------------------------------------------------------

    def _build_game(self):
        frame = tk.Frame(self.container, bg=self.BG_COLOR)
        
        # Left side - board with player names
        board_container = tk.Frame(frame, bg=self.BG_COLOR)
        board_container.pack(side="left", padx=(20, 5), pady=10)
        
        # Top player name (Black or opponent)
        self.top_player_label = tk.Label(board_container, text="Black", font=("Arial", 12, "bold"),
                                         bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.top_player_label.pack(anchor="w", pady=(0, 5))
        
        self.canvas = tk.Canvas(board_container, width=self.BOARD_SIZE, height=self.BOARD_SIZE, 
                               bg=self.BG_COLOR, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_board_click)
        
        # Bottom player name (White or player)
        self.bottom_player_label = tk.Label(board_container, text="White", font=("Arial", 12, "bold"),
                                            bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.bottom_player_label.pack(anchor="w", pady=(5, 0))
        
        # Vertical Evaluation Bar (between board and sidebar)
        self.eval_bar_frame = tk.Frame(frame, bg=self.BG_COLOR)
        self.eval_bar_frame.pack(side="left", fill="y", padx=(5, 10), pady=30)
        
        # Eval bar canvas - 20px wide, same height as board
        self.eval_bar_canvas = tk.Canvas(self.eval_bar_frame, width=20, height=self.BOARD_SIZE,
                                         bg="#333", highlightthickness=1, highlightbackground="#555")
        self.eval_bar_canvas.pack()
        
        # Eval score label below bar
        self.eval_score_label = tk.Label(self.eval_bar_frame, text="0.0", font=("Consolas", 10),
                                         bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.eval_score_label.pack(pady=(5, 0))
        
        # Initially hide eval bar
        self.eval_bar_frame.pack_forget()
        
        # Right side - sidebar
        sidebar = tk.Frame(frame, bg=self.PANEL_COLOR, width=250)
        sidebar.pack(side="right", fill="y", padx=0)
        
        # In-game settings gear at top right of sidebar (use Label for transparent look)
        settings_row = tk.Frame(sidebar, bg=self.PANEL_COLOR)
        settings_row.pack(fill="x", padx=5, pady=5)
        gear_label = tk.Label(settings_row, text="⚙", font=("Arial", 14), bg=self.PANEL_COLOR, 
                              fg=self.TEXT_COLOR, cursor="hand2")
        gear_label.pack(side="right")
        gear_label.bind("<Button-1>", lambda e: self._open_game_settings())
        
        self.turn_label = tk.Label(sidebar, text="White to Move", font=("Arial", 14, "bold"), 
                                  bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.turn_label.pack(pady=10)
        
        # Evaluation display (hidden by default)
        self.eval_frame = tk.Frame(sidebar, bg=self.PANEL_COLOR)
        self.eval_label = tk.Label(self.eval_frame, text="Eval: 0.00", font=("Consolas", 12),
                                   bg=self.PANEL_COLOR, fg="#81B64C")
        self.eval_label.pack()
        
        tk.Label(sidebar, text="Move History", bg=self.PANEL_COLOR, fg="#888").pack(anchor="w", padx=10)
        self.history_text = tk.Text(sidebar, height=12, width=25, bg="#3A3836", fg="white", 
                                   font=("Consolas", 10), state="disabled", relief="flat")
        self.history_text.pack(padx=10, pady=5)
        
        controls = tk.Frame(sidebar, bg=self.PANEL_COLOR)
        controls.pack(side="bottom", fill="x", pady=20, padx=10)
        tk.Button(controls, text="Back to Menu", command=self._end_game, 
                 bg="#555", fg="white", relief="flat").pack(fill="x", pady=5)
        tk.Button(controls, text="Export", command=self._open_export_dialog, 
                 bg="#555", fg="white", relief="flat").pack(fill="x", pady=5)
                 
        return frame
    
    def _open_game_settings(self):
        """Open in-game settings dialog - changes only apply when OK is clicked."""
        dialog = ttk.Toplevel(self.root)
        dialog.title("Game Settings")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.geometry("280x150")
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Temporary variable
        temp_show_eval = tk.BooleanVar(value=self.show_evaluation.get())
        
        ttk.Label(dialog, text="Game Settings", font=("Arial", 12, "bold")).pack(pady=(15, 15))
        
        # Show evaluation toggle
        eval_frame = ttk.Frame(dialog)
        eval_frame.pack(fill="x", padx=20, pady=10)
        ttk.Checkbutton(eval_frame, text="Show Evaluation", variable=temp_show_eval,
                        bootstyle="success-round-toggle").pack(anchor="w")
        
        def apply_and_close():
            # Only trigger toggle if value changed
            if temp_show_eval.get() != self.show_evaluation.get():
                self.show_evaluation.set(temp_show_eval.get())
                self._toggle_eval_display()
            dialog.destroy()
        
        ttk.Button(dialog, text="OK", command=apply_and_close, 
                   bootstyle="success", width=10).pack(pady=15)
        
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    
    def _toggle_eval_display(self):
        """Toggle evaluation bar visibility and start/stop eval engine."""
        if self.show_evaluation.get():
            # Show eval bar
            self.eval_bar_frame.pack(side="left", fill="y", padx=(5, 10), pady=30, 
                                     after=self.canvas.master)
            # Start eval engine if not running
            self._start_eval_engine()
            # Trigger initial evaluation
            self._request_evaluation()
        else:
            # Hide eval bar
            self.eval_bar_frame.pack_forget()
            # Stop eval engine
            self._stop_eval_engine()
    
    def _start_eval_engine(self):
        """Start dedicated Stockfish engine for evaluation."""
        if self.eval_engine is None:
            eval_path = os.path.join(ENGINES_DIR, "stockfish-windows-x86-64-avx2.exe")
            if os.path.exists(eval_path):
                self.eval_engine = UCIEngine(eval_path, "EvalEngine")
                self.eval_engine.set_option("Use NNUE", "true")
                self.eval_engine.set_option("Threads", "1")  # Use 1 thread for eval
                self.eval_engine.new_game()
            else:
                print(f"Eval engine not found: {eval_path}")
    
    def _stop_eval_engine(self):
        """Stop the evaluation engine."""
        if self.eval_engine:
            self.eval_engine.quit()
            self.eval_engine = None
    
    def _request_evaluation(self):
        """Request position evaluation from eval engine."""
        if not self.show_evaluation.get() or not self.eval_engine:
            return
        
        def evaluate():
            if self.eval_engine and self.game_running:
                fen = self.board.fen()
                # Get evaluation using info score
                self.eval_engine._send(f"position fen {fen}")
                self.eval_engine._send(f"go depth {self.eval_depth}")
                
                score_cp = 0
                score_mate = None
                
                start = time.time()
                while time.time() - start < 10:  # 10 second timeout
                    if not self.eval_engine:  # Engine was stopped
                        return
                    line = self.eval_engine._get_line(timeout=0.5)
                    if line:
                        if "score cp" in line:
                            # Extract centipawn score
                            parts = line.split()
                            try:
                                idx = parts.index("cp") + 1
                                score_cp = int(parts[idx])
                            except:
                                pass
                        elif "score mate" in line:
                            # Mate score
                            parts = line.split()
                            try:
                                idx = parts.index("mate") + 1
                                score_mate = int(parts[idx])
                            except:
                                pass
                        elif line.startswith("bestmove"):
                            break
                
                # Convert to pawns (centipawns / 100)
                if score_mate is not None:
                    # Mate scores: positive = white wins, negative = black wins
                    if score_mate > 0:
                        self.current_eval = 100.0  # White winning
                    else:
                        self.current_eval = -100.0  # Black winning
                else:
                    self.current_eval = score_cp / 100.0
                
                # Adjust for side to move (scores are from side to move perspective)
                if self.board.turn == chess.BLACK:
                    self.current_eval = -self.current_eval
                
                # Update UI on main thread
                self.root.after(0, self._update_eval_bar)
        
        threading.Thread(target=evaluate, daemon=True).start()
    
    def _update_eval_bar(self):
        """Update the visual evaluation bar with smooth animation."""
        # Animate towards target value
        diff = self.current_eval - self.display_eval
        if abs(diff) > 0.01:
            # Smooth interpolation (ease-out, slower = smoother)
            self.display_eval += diff * 0.15
            # Schedule next frame (~60fps)
            self.root.after(16, self._update_eval_bar)
        else:
            self.display_eval = self.current_eval
        
        self.eval_bar_canvas.delete("all")
        
        # Clamp eval to reasonable display range (-10 to +10)
        display_val = max(-10, min(10, self.display_eval))
        
        # Calculate white portion (0.5 = equal, 1.0 = white winning, 0.0 = black winning)
        white_ratio = 0.5 + (display_val / 20.0)
        white_ratio = max(0.05, min(0.95, white_ratio))
        
        bar_height = self.BOARD_SIZE
        bar_width = 20
        
        # Draw black portion (top)
        black_height = int(bar_height * (1 - white_ratio))
        self.eval_bar_canvas.create_rectangle(0, 0, bar_width, black_height, 
                                               fill="#333", outline="")
        
        # Draw white portion (bottom)
        self.eval_bar_canvas.create_rectangle(0, black_height, bar_width, bar_height,
                                               fill="#EEE", outline="")
        
        # Draw center line
        center_y = bar_height // 2
        self.eval_bar_canvas.create_line(0, center_y, bar_width, center_y, 
                                          fill="#666", width=2)
        
        # Update score label with current eval (not display_eval for accuracy)
        if abs(self.current_eval) >= 100:
            # Mate
            if self.current_eval > 0:
                self.eval_score_label.config(text="M+", fg="#81B64C")
            else:
                self.eval_score_label.config(text="M-", fg="#E74C3C")
        else:
            sign = "+" if self.current_eval > 0 else ""
            self.eval_score_label.config(text=f"{sign}{self.current_eval:.1f}",
                                         fg="#81B64C" if self.current_eval > 0.5 else 
                                         "#E74C3C" if self.current_eval < -0.5 else "#AAA")
    
    def _update_player_labels(self):
        """Update the player name labels on the board."""
        # Top label shows black player (opponent from white's perspective)
        self.top_player_label.config(text=f"⬛ {self.black_player_name}")
        # Bottom label shows white player
        self.bottom_player_label.config(text=f"⬜ {self.white_player_name}")

    def _show_game(self):
        self.menu_frame.pack_forget()
        self.game_frame.pack(fill="both", expand=True)
        self._update_board()

    # --------------------------------------------------------------------------
    # GAME LOGIC
    # --------------------------------------------------------------------------

    def _start_game(self):
        self._stop_engines()
        self.game_mode = self.mode_var.get()
        self.board = chess.Board()
        self.game_running = True
        self.current_eval = 0.0
        
        if self.game_mode == "PvP":
            # Player vs Player
            self.white_player_name = "Player 1"
            self.black_player_name = "Player 2"
        
        elif self.game_mode == "PvC":
            self.player_side = chess.WHITE if self.pvc_side_var.get() == "White" else chess.BLACK
            selected_engine = self.pvc_engine_var.get()
            if not selected_engine:
                messagebox.showerror("Error", "Please select an engine!")
                return
            # Try .exe first, then .bat
            eng_path = os.path.join(ENGINES_DIR, selected_engine + ".exe")
            if not os.path.exists(eng_path):
                eng_path = os.path.join(ENGINES_DIR, selected_engine + ".bat")
            if not os.path.exists(eng_path):
                messagebox.showerror("Error", f"Engine not found: {selected_engine}")
                return
            eng = UCIEngine(eng_path, selected_engine)
            # Enable NNUE by default (if supported, silently ignored if not)
            eng.set_option("Use NNUE", "true")
            # Configure engine settings
            if self.ponder_enabled.get():
                eng.set_option("Ponder", "true")
            eng.set_option("Skill Level", self.engine_skill.get())
            eng.set_option("Contempt", self.engine_contempt.get())
            eng.new_game()
            self.engines['computer'] = eng
            
            # Set player names
            if self.player_side == chess.WHITE:
                self.white_player_name = "Player"
                self.black_player_name = selected_engine
            else:
                self.white_player_name = selected_engine
                self.black_player_name = "Player"

        elif self.game_mode == "CvC":
            white_name = self.white_engine_var.get()
            black_name = self.black_engine_var.get()
            if not white_name or not black_name:
                messagebox.showerror("Error", "Select two engines!")
                return
            # Find engine paths (try .exe first, then .bat)
            def get_engine_path(name):
                exe_path = os.path.join(ENGINES_DIR, name + ".exe")
                if os.path.exists(exe_path):
                    return exe_path
                bat_path = os.path.join(ENGINES_DIR, name + ".bat")
                if os.path.exists(bat_path):
                    return bat_path
                return None
            w_path = get_engine_path(white_name)
            b_path = get_engine_path(black_name)
            if not w_path or not b_path:
                messagebox.showerror("Error", "Engine not found!")
                return
            w_eng = UCIEngine(w_path, white_name)
            b_eng = UCIEngine(b_path, black_name)
            if not w_eng.process or not b_eng.process:
                 messagebox.showerror("Error", "Failed to start engines.")
                 return
            # Apply settings to both engines (including NNUE)
            for eng in [w_eng, b_eng]:
                eng.set_option("Use NNUE", "true")
                eng.set_option("Skill Level", self.engine_skill.get())
                eng.set_option("Contempt", self.engine_contempt.get())
                eng.new_game()
            self.engines[chess.WHITE] = w_eng
            self.engines[chess.BLACK] = b_eng
            
            # Set player names
            self.white_player_name = white_name
            self.black_player_name = black_name
            
            threading.Thread(target=self._cvc_loop, daemon=True).start()
        
        # Update player name labels
        self._update_player_labels()
        self._show_game()
        if self.game_mode == "PvC" and self.player_side == chess.BLACK:
            self.root.after(500, self._pvc_computer_move)

    # _find_engine_path is no longer needed - engine selection is handled via dropdown

    def _stop_engines(self):
        self.game_running = False
        for eng in self.engines.values():
            eng.quit()
        self.engines.clear()
        # Also stop eval engine
        self._stop_eval_engine()

    def _end_game(self):
        self._stop_engines()
        # Reset and hide eval bar
        self.show_evaluation.set(False)
        self.eval_bar_frame.pack_forget()
        self._show_menu()

    def _open_export_dialog(self):
        """Open export dialog with PGN/FEN selection, preview, and copy."""
        dialog = ttk.Toplevel(self.root)
        dialog.title("Export")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.geometry("450x400")
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        ttk.Label(dialog, text="Export Game", font=("Arial", 14, "bold")).pack(pady=(15, 10))
        
        # Format selection
        format_var = tk.StringVar(value="pgn")
        format_frame = ttk.Frame(dialog)
        format_frame.pack(fill="x", padx=20, pady=5)
        ttk.Label(format_frame, text="Format:").pack(side="left")
        ttk.Radiobutton(format_frame, text="PGN", variable=format_var, value="pgn",
                        command=lambda: update_preview()).pack(side="left", padx=10)
        ttk.Radiobutton(format_frame, text="FEN", variable=format_var, value="fen",
                        command=lambda: update_preview()).pack(side="left", padx=10)
        
        # Preview label
        ttk.Label(dialog, text="Preview:").pack(anchor="w", padx=20, pady=(10, 5))
        
        # Preview text area
        preview_frame = ttk.Frame(dialog)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=5)
        preview_text = tk.Text(preview_frame, height=12, width=50, bg="#3A3836", fg="white",
                               font=("Consolas", 9), relief="flat", wrap="word")
        preview_text.pack(fill="both", expand=True)
        
        def get_pgn_string():
            game = chess.pgn.Game()
            game.headers["Event"] = "Chess App Game"
            game.headers["Site"] = "Local"
            game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = "1"
            if self.game_mode == "CvC":
                game.headers["White"] = self.engines.get(chess.WHITE, type('', (), {'name': 'White'})()).name if chess.WHITE in self.engines else "White"
                game.headers["Black"] = self.engines.get(chess.BLACK, type('', (), {'name': 'Black'})()).name if chess.BLACK in self.engines else "Black"
            elif self.game_mode == "PvC":
                game.headers["White"] = "Player" if self.player_side == chess.WHITE else "Computer"
                game.headers["Black"] = "Computer" if self.player_side == chess.WHITE else "Player"
            else:
                game.headers["White"] = "Player 1"
                game.headers["Black"] = "Player 2"
            game.headers["Result"] = self.board.result()
            node = game
            for move in self.board.move_stack:
                node = node.add_variation(move)
            return str(game)
        
        def update_preview():
            preview_text.config(state="normal")
            preview_text.delete("1.0", "end")
            if format_var.get() == "pgn":
                preview_text.insert("1.0", get_pgn_string())
            else:
                preview_text.insert("1.0", self.board.fen())
            preview_text.config(state="disabled")
        
        def copy_to_clipboard():
            content = preview_text.get("1.0", "end").strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            copy_btn.config(text="Copied!")
            dialog.after(1500, lambda: copy_btn.config(text="Copy"))
        
        def export_file():
            import os
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            if format_var.get() == "pgn":
                path = filedialog.asksaveasfilename(
                    initialdir=desktop,
                    defaultextension=".pgn",
                    filetypes=[("PGN Files", "*.pgn"), ("All Files", "*.*")]
                )
                if path:
                    with open(path, "w") as f:
                        f.write(get_pgn_string())
                    messagebox.showinfo("Export", f"Game saved to {path}")
                    dialog.destroy()
            else:
                path = filedialog.asksaveasfilename(
                    initialdir=desktop,
                    defaultextension=".fen",
                    filetypes=[("FEN Files", "*.fen"), ("Text Files", "*.txt"), ("All Files", "*.*")]
                )
                if path:
                    with open(path, "w") as f:
                        f.write(self.board.fen())
                    messagebox.showinfo("Export", f"FEN saved to {path}")
                    dialog.destroy()
        
        # Initial preview
        update_preview()
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        copy_btn = ttk.Button(btn_frame, text="Copy", command=copy_to_clipboard, bootstyle="info", width=10)
        copy_btn.pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Export", command=export_file, bootstyle="success", width=10).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, bootstyle="secondary", width=10).pack(side="left", padx=5)

    # --------------------------------------------------------------------------
    # RENDER & INTERACTION
    # --------------------------------------------------------------------------

    def _update_board(self):
        self.canvas.delete("all")
        last_move = self.board.peek() if self.board.move_stack else None
        
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                x = file * self.SQUARE_SIZE
                y = rank * self.SQUARE_SIZE
                is_light = (file + rank) % 2 == 0
                bg = self.LIGHT_COLOR if is_light else self.DARK_COLOR
                
                if sq == self.selected_square: bg = self.SELECTED_COLOR
                elif last_move and (sq == last_move.from_square or sq == last_move.to_square):
                    bg = self.LAST_MOVE_COLOR
                
                self.canvas.create_rectangle(x, y, x + self.SQUARE_SIZE, y + self.SQUARE_SIZE, 
                                           fill=bg, outline="")
                
                if sq in self.legal_moves:
                    r = self.SQUARE_SIZE // 6
                    cx, cy = x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2
                    self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#888888", outline="")

                piece = self.board.piece_at(sq)
                if piece:
                    img = self.pieces_images.get(piece.symbol())
                    if img: self.canvas.create_image(x + self.SQUARE_SIZE//2, y + self.SQUARE_SIZE//2, image=img)
                    else:
                        txt = piece.unicode_symbol()
                        self.canvas.create_text(x + self.SQUARE_SIZE//2, y + self.SQUARE_SIZE//2, 
                                              text=txt, font=("Arial", 30))

        self.history_text.config(state="normal")
        self.history_text.delete(1.0, tk.END)
        san_moves = []
        temp_board = chess.Board()
        for m in self.board.move_stack:
            san = temp_board.san(m)
            san_moves.append(san)
            temp_board.push(m)
        hist_str = ""
        for i in range(0, len(san_moves), 2):
            w = san_moves[i]
            b = san_moves[i+1] if i+1 < len(san_moves) else ""
            hist_str += f"{i//2 + 1}. {w} {b}\n"
        self.history_text.insert("end", hist_str)
        self.history_text.see("end")
        self.history_text.config(state="disabled")
        
        if self.board.is_game_over(): self.turn_label.config(text=f"Game Over: {self.board.result()}")
        else: self.turn_label.config(text="White's Turn" if self.board.turn == chess.WHITE else "Black's Turn")

    def _animate_move(self, move, callback):
        start_sq = move.from_square
        end_sq = move.to_square
        
        start_col, start_row = chess.square_file(start_sq), 7 - chess.square_rank(start_sq)
        end_col, end_row = chess.square_file(end_sq), 7 - chess.square_rank(end_sq)
        
        start_x = start_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
        start_y = start_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
        end_x = end_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
        end_y = end_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
        
        piece = self.board.piece_at(start_sq)
        if not piece:
            callback()
            return
            
        img = self.pieces_images.get(piece.symbol())
        if img: floater = self.canvas.create_image(start_x, start_y, image=img, tags="anim")
        else: floater = self.canvas.create_text(start_x, start_y, text=piece.unicode_symbol(), font=("Arial", 30), tags="anim")

        bg_col = self.LIGHT_COLOR if (start_col + start_row) % 2 == 0 else self.DARK_COLOR
        if start_sq == self.selected_square: bg_col = self.SELECTED_COLOR
        rect_x = start_col * self.SQUARE_SIZE
        rect_y = start_row * self.SQUARE_SIZE
        cover = self.canvas.create_rectangle(rect_x, rect_y, rect_x + self.SQUARE_SIZE, rect_y + self.SQUARE_SIZE, 
                                     fill=bg_col, outline="")
        self.canvas.tag_raise(floater)

        steps = 20
        duration_ms = 150 
        step_time = max(5, duration_ms // steps)
        
        def ease_in_out(t): return t * t * (3 - 2 * t)
        
        def step(i):
            if i <= steps:
                t = i / steps
                f = ease_in_out(t)
                cx = start_x + (end_x - start_x) * f
                cy = start_y + (end_y - start_y) * f
                self.canvas.coords(floater, cx, cy)
                self.root.after(step_time, lambda: step(i+1))
            else:
                self.canvas.delete(floater)
                self.canvas.delete(cover)
                callback()
        step(0)

    def _on_board_click(self, event):
        if not self.game_running or self.game_mode == "CvC": return
        if self.game_mode == "PvC" and self.board.turn != self.player_side: return

        col = event.x // self.SQUARE_SIZE
        row = event.y // self.SQUARE_SIZE
        sq = chess.square(col, 7 - row)
        
        if self.selected_square is not None:
            move = chess.Move(self.selected_square, sq)
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                 if (self.board.turn == chess.WHITE and chess.square_rank(sq) == 7) or \
                    (self.board.turn == chess.BLACK and chess.square_rank(sq) == 0):
                    move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)

            valid = False
            if move in self.board.legal_moves: valid = True
            else:
                move_q = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
                if move_q in self.board.legal_moves:
                    move = move_q
                    valid = True

            if valid:
                self.selected_square = None
                self.legal_moves = []
                self._animate_move(move, lambda: self._finalize_move(move))
                return
            
            if self.board.piece_at(sq) and self.board.piece_at(sq).color == self.board.turn:
                 self.selected_square = sq
                 self.legal_moves = [m.to_square for m in self.board.legal_moves if m.from_square == sq]
            else:
                self.selected_square = None
                self.legal_moves = []
        else:
            if self.board.piece_at(sq) and self.board.piece_at(sq).color == self.board.turn:
                self.selected_square = sq
                self.legal_moves = [m.to_square for m in self.board.legal_moves if m.from_square == sq]
        self._update_board()

    def _finalize_move(self, move):
        """Finalize player's move and trigger computer response."""
        engine = self.engines.get('computer')
        
        # Check if engine was pondering on this move
        if engine and engine.is_pondering and self.ponder_enabled.get():
            if engine.ponder_move == move.uci():
                # Ponderhit - engine predicted correctly!
                self.board.push(move)
                self._update_board()
                self._request_evaluation()  # Update eval after move
                if not self.board.is_game_over() and self.game_mode == "PvC":
                    def get_ponderhit_move():
                        best, ponder = engine.ponderhit()
                        if best:
                            self.root.after(0, lambda: self._apply_move_uci_with_ponder(best, ponder))
                    threading.Thread(target=get_ponderhit_move, daemon=True).start()
                return
            else:
                # Wrong prediction - stop and recalculate
                engine.stop_pondering()
        
        self.board.push(move)
        self._update_board()
        self._request_evaluation()  # Update eval after move
        if not self.board.is_game_over() and self.game_mode == "PvC":
             self.root.after(100, self._pvc_computer_move)

    def _pvc_computer_move(self):
        if not self.game_running or self.board.is_game_over(): return
        def think():
            engine = self.engines.get('computer')
            if engine:
                depth = self.engine_depth.get()
                best, ponder = engine.get_best_move(self.board.fen(), depth)
                if best:
                    self.root.after(0, lambda: self._apply_move_uci_with_ponder(best, ponder))
        threading.Thread(target=think, daemon=True).start()

    def _apply_move_uci(self, uci):
        if not self.game_running: return
        move = chess.Move.from_uci(uci)
        # Verify move is legal before animating
        if move not in self.board.legal_moves:
            print(f"Ignoring illegal move: {uci}")
            self.cvc_move_done.set()  # Signal anyway to prevent deadlock
            return
        self._animate_move(move, lambda: self._finalize_move_uci(move))
    
    def _apply_move_uci_with_ponder(self, uci, ponder_move):
        """Apply move and optionally start pondering."""
        if not self.game_running: return
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            print(f"Ignoring illegal move: {uci}")
            return
        self._animate_move(move, lambda: self._finalize_move_uci_with_ponder(move, ponder_move))

    def _finalize_move_uci(self, move):
        # Double check move is still legal
        if move not in self.board.legal_moves:
            print(f"Move no longer legal: {move}")
            self.cvc_move_done.set()
            return
        self.board.push(move)
        self._update_board()
        self._request_evaluation()  # Update eval after move
        # Signal CvC loop that move is done
        self.cvc_move_done.set()
    
    def _finalize_move_uci_with_ponder(self, move, ponder_move):
        """Finalize engine move and start pondering if enabled."""
        if move not in self.board.legal_moves:
            print(f"Move no longer legal: {move}")
            return
        self.board.push(move)
        self._update_board()
        self._request_evaluation()  # Update eval after move
        
        # Start pondering if enabled and we have a ponder move
        if self.ponder_enabled.get() and ponder_move and not self.board.is_game_over():
            engine = self.engines.get('computer')
            if engine:
                engine.start_pondering(self.board.fen(), ponder_move)
    
    def _cvc_loop(self):
        while self.game_running:
            try:
                if self.board.is_game_over():
                    break
                engine = self.engines.get(self.board.turn)
                if not engine:
                    break
                depth = self.engine_depth.get()
                fen = self.board.fen()  # Get FEN before potential race
                best, _ = engine.get_best_move(fen, depth)
                if not self.game_running: 
                    break
                if best:
                    # Clear event and schedule move
                    self.cvc_move_done.clear()
                    self.root.after(0, lambda u=best: self._apply_move_uci(u))
                    # Wait for move to complete (with timeout)
                    self.cvc_move_done.wait(timeout=5.0)
                    time.sleep(0.5)  # Small delay between moves
                else: 
                    break
            except Exception as e:
                print(f"CvC loop error: {e}")
                break
    
    def run(self):
        self.root.mainloop()
        self._stop_engines()

if __name__ == "__main__":
    app = ChessApp()
    app.run()
