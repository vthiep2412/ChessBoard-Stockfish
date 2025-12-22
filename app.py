import tkinter as tk
from tkinter import messagebox
import chess
import subprocess
import threading
import os
import time

# ==============================================================================
# STOCKFISH ENGINE
# ==============================================================================

class StockfishEngine:
    def __init__(self, path):
        self.process = subprocess.Popen(
            path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name Skill Level value 20")
        self._send("isready")
        self._wait_for("readyok")
    
    def _send(self, cmd):
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
    
    def _wait_for(self, keyword, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            line = self.process.stdout.readline().strip()
            if keyword in line:
                return line
        return None
    
    def get_best_move(self, fen, time_ms=1000):
        self._send(f"position fen {fen}")
        self._send(f"go movetime {time_ms}")
        while True:
            line = self.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                return line.split()[1]
    
    def set_skill(self, level):
        self._send(f"setoption name Skill Level value {level}")
    
    def quit(self):
        self._send("quit")
        self.process.terminate()

# ==============================================================================
# CHESS GUI - DARK MODE
# ==============================================================================

class ChessGUI:
    SQUARE_SIZE = 70
    
    # Dark mode colors (chess.com style)
    LIGHT_COLOR = "#779556"
    DARK_COLOR = "#425035"
    SELECTED_COLOR = "#F7EC45"
    LEGAL_COLOR = "#B0D165"
    LAST_MOVE_LIGHT = "#B9CA43"
    LAST_MOVE_DARK = "#8A9A2B"
    BG_COLOR = "#312E2B"
    PANEL_COLOR = "#272522"
    TEXT_COLOR = "#BABABA"
    
    PIECES = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chess - Stockfish")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG_COLOR)
        
        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = set()
        self.vs_engine = True
        self.player_color = chess.WHITE
        self.engine = None
        self.think_time = 1000
        
        # Find stockfish
        self.stockfish_path = self._find_stockfish()
        if self.stockfish_path:
            try:
                self.engine = StockfishEngine(self.stockfish_path)
            except Exception as e:
                print(f"Engine error: {e}")
        
        self._create_ui()
        self._draw_board()
    
    def _find_stockfish(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for f in os.listdir(script_dir):
            if "stockfish" in f.lower() and f.endswith(".exe"):
                return os.path.join(script_dir, f)
        return None
    
    def _create_ui(self):
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(padx=10, pady=10)
        
        # Board canvas
        board_size = self.SQUARE_SIZE * 8
        self.canvas = tk.Canvas(main_frame, width=board_size, height=board_size,
                               highlightthickness=0, bg=self.BG_COLOR)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self._on_click)
        
        # Side panel
        side = tk.Frame(main_frame, padx=20, bg=self.PANEL_COLOR)
        side.pack(side=tk.LEFT, fill=tk.Y)
        
        # Turn label
        self.turn_label = tk.Label(side, text="White's Turn", font=("Arial", 16, "bold"),
                                   bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.turn_label.pack(pady=10)
        
        # Mode selection
        mode_frame = tk.LabelFrame(side, text="Play Mode", padx=10, pady=10,
                                   bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.mode_var = tk.StringVar(value="vs_engine")
        for text, val in [("vs Engine", "vs_engine"), ("2 Player", "2_player")]:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=val,
                          command=self._mode_changed, bg=self.PANEL_COLOR, 
                          fg=self.TEXT_COLOR, selectcolor=self.BG_COLOR,
                          activebackground=self.PANEL_COLOR).pack(anchor=tk.W)
        
        # Color selection
        self.color_frame = tk.LabelFrame(side, text="Play as", padx=10, pady=10,
                                         bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.color_frame.pack(fill=tk.X, pady=5)
        
        self.color_var = tk.StringVar(value="white")
        for text, val in [("White", "white"), ("Black", "black")]:
            tk.Radiobutton(self.color_frame, text=text, variable=self.color_var, value=val,
                          command=self._color_changed, bg=self.PANEL_COLOR,
                          fg=self.TEXT_COLOR, selectcolor=self.BG_COLOR,
                          activebackground=self.PANEL_COLOR).pack(anchor=tk.W)
        
        # Skill slider
        skill_frame = tk.LabelFrame(side, text="Engine Skill (0-20)", padx=10, pady=10,
                                    bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        skill_frame.pack(fill=tk.X, pady=5)
        
        self.skill_var = tk.IntVar(value=20)
        tk.Scale(skill_frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.skill_var,
                command=self._skill_changed, bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                highlightthickness=0, troughcolor=self.BG_COLOR).pack(fill=tk.X)
        
        # Think time slider
        time_frame = tk.LabelFrame(side, text="Think Time (seconds)", padx=10, pady=10,
                                   bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        time_frame.pack(fill=tk.X, pady=5)
        
        self.time_var = tk.DoubleVar(value=1.0)
        tk.Scale(time_frame, from_=0.5, to=5.0, resolution=0.5, orient=tk.HORIZONTAL,
                variable=self.time_var, command=self._time_changed,
                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                highlightthickness=0, troughcolor=self.BG_COLOR).pack(fill=tk.X)
        
        # Buttons
        btn_frame = tk.Frame(side, bg=self.PANEL_COLOR)
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="↩️ Undo", command=self._undo, width=10,
                 bg="#555", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="🆕 New", command=self._new_game, width=10,
                 bg="#555", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        
        # Engine move button
        self.engine_btn = tk.Button(side, text="🤖 Engine Move", command=self._engine_move,
                                    bg="#81B64C", fg="white", relief=tk.FLAT)
        self.engine_btn.pack(fill=tk.X, pady=5)
        self.engine_btn.pack_forget()
        
        # Status
        self.status_label = tk.Label(side, text="", font=("Arial", 10),
                                     bg=self.PANEL_COLOR, fg="#81B64C")
        self.status_label.pack(pady=5)
        
        # History
        history_frame = tk.LabelFrame(side, text="Moves", padx=5, pady=5,
                                      bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.history_text = tk.Text(history_frame, width=20, height=8, font=("Courier", 10),
                                    bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.history_text.pack(fill=tk.BOTH, expand=True)
    
    def _draw_board(self):
        self.canvas.delete("all")
        
        last_from, last_to = None, None
        if self.board.move_stack:
            lm = self.board.move_stack[-1]
            last_from, last_to = lm.from_square, lm.to_square
        
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                x1, y1 = file * self.SQUARE_SIZE, (7 - rank) * self.SQUARE_SIZE
                x2, y2 = x1 + self.SQUARE_SIZE, y1 + self.SQUARE_SIZE
                
                is_light = (file + rank) % 2 == 1
                
                if sq == self.selected_square:
                    color = self.SELECTED_COLOR
                elif sq in self.legal_targets:
                    color = self.LEGAL_COLOR
                elif sq == last_from or sq == last_to:
                    color = self.LAST_MOVE_LIGHT if is_light else self.LAST_MOVE_DARK
                else:
                    color = self.LIGHT_COLOR if is_light else self.DARK_COLOR
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                piece = self.board.piece_at(sq)
                if piece:
                    pc = self.PIECES.get(piece.symbol(), "?")
                    tc = "#000" if piece.color == chess.BLACK else "#FFF"
                    self.canvas.create_text(x1 + 35, y1 + 35, text=pc, font=("Arial", 45), fill=tc)
        
        turn = "White" if self.board.turn else "Black"
        self.turn_label.config(text=f"{turn}'s Turn")
        self._update_history()
        
        if self.board.is_game_over():
            r = self.board.result()
            msg = "White wins!" if r == "1-0" else "Black wins!" if r == "0-1" else "Draw!"
            messagebox.showinfo("Game Over", msg)
    
    def _on_click(self, event):
        if self.board.is_game_over():
            return
        if self.vs_engine and self.board.turn != self.player_color:
            return
        
        file = event.x // self.SQUARE_SIZE
        rank = 7 - (event.y // self.SQUARE_SIZE)
        
        if 0 <= file < 8 and 0 <= rank < 8:
            self._handle_click(chess.square(file, rank))
    
    def _handle_click(self, sq):
        piece = self.board.piece_at(sq)
        
        if self.selected_square is None:
            if piece and piece.color == self.board.turn:
                self.selected_square = sq
                self.legal_targets = {m.to_square for m in self.board.legal_moves if m.from_square == sq}
                self._draw_board()
        else:
            if sq in self.legal_targets:
                move = chess.Move(self.selected_square, sq)
                if self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                    if chess.square_rank(sq) in [0, 7]:
                        move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
                
                self.board.push(move)
                self.selected_square = None
                self.legal_targets = set()
                self._draw_board()
                
                if self.vs_engine and not self.board.is_game_over():
                    self.root.after(100, self._engine_respond)
            else:
                if piece and piece.color == self.board.turn:
                    self.selected_square = sq
                    self.legal_targets = {m.to_square for m in self.board.legal_moves if m.from_square == sq}
                else:
                    self.selected_square = None
                    self.legal_targets = set()
                self._draw_board()
    
    def _engine_respond(self):
        if not self.engine:
            self.status_label.config(text="No engine!")
            return
        
        self.status_label.config(text="Thinking...")
        self.root.update()
        
        def think():
            uci = self.engine.get_best_move(self.board.fen(), self.think_time)
            self.root.after(0, lambda: self._make_engine_move(uci))
        
        threading.Thread(target=think, daemon=True).start()
    
    def _make_engine_move(self, uci):
        if uci:
            self.board.push(chess.Move.from_uci(uci))
            self._draw_board()
        self.status_label.config(text="")
    
    def _engine_move(self):
        if not self.board.is_game_over():
            self._engine_respond()
    
    def _undo(self):
        if self.board.move_stack:
            self.board.pop()
            if self.vs_engine and self.board.move_stack:
                self.board.pop()
            self.selected_square = None
            self.legal_targets = set()
            self._draw_board()
    
    def _new_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = set()
        self._draw_board()
        if self.vs_engine and self.player_color == chess.BLACK:
            self.root.after(500, self._engine_respond)
    
    def _mode_changed(self):
        self.vs_engine = self.mode_var.get() == "vs_engine"
        if self.vs_engine:
            self.color_frame.pack(fill=tk.X, pady=5, after=self.color_frame.master.winfo_children()[1])
            self.engine_btn.pack_forget()
        else:
            self.color_frame.pack_forget()
            self.engine_btn.pack(fill=tk.X, pady=5)
    
    def _color_changed(self):
        self.player_color = chess.WHITE if self.color_var.get() == "white" else chess.BLACK
    
    def _skill_changed(self, val):
        if self.engine:
            self.engine.set_skill(int(float(val)))
    
    def _time_changed(self, val):
        self.think_time = int(float(val) * 1000)
    
    def _update_history(self):
        self.history_text.delete(1.0, tk.END)
        moves = list(self.board.move_stack)
        for i in range(0, len(moves), 2):
            w = moves[i].uci()
            b = moves[i + 1].uci() if i + 1 < len(moves) else ""
            self.history_text.insert(tk.END, f"{i//2+1}. {w} {b}\n")
    
    def run(self):
        if self.vs_engine and self.player_color == chess.BLACK:
            self.root.after(500, self._engine_respond)
        self.root.mainloop()
        if self.engine:
            self.engine.quit()

if __name__ == "__main__":
    ChessGUI().run()
