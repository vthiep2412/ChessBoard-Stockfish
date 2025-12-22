import streamlit as st
import chess
import chess.svg
import base64
import time
import os

# ==============================================================================
# STOCKFISH ENGINE WRAPPER
# ==============================================================================

from stockfish import Stockfish

def get_stockfish_path():
    """Get the path to stockfish.exe in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stockfish_path = os.path.join(script_dir, "stockfish.exe")
    
    # Also check for stockfish-windows-x86-64-avx2.exe (common download name)
    if not os.path.exists(stockfish_path):
        for f in os.listdir(script_dir):
            if f.lower().startswith("stockfish") and f.lower().endswith(".exe"):
                stockfish_path = os.path.join(script_dir, f)
                break
    
    return stockfish_path

@st.cache_resource
def init_stockfish(depth=15, skill_level=20):
    """Initialize Stockfish engine (cached to avoid reloading)."""
    stockfish_path = get_stockfish_path()
    
    if not os.path.exists(stockfish_path):
        return None
    
    try:
        sf = Stockfish(
            path=stockfish_path,
            depth=depth,
            parameters={
                "Threads": 2,
                "Hash": 128,
                "Skill Level": skill_level
            }
        )
        return sf
    except Exception as e:
        st.error(f"Failed to initialize Stockfish: {e}")
        return None

def get_best_move(engine, board, think_time_ms=1000):
    """Get the best move from Stockfish for the current position."""
    if engine is None:
        return None
    
    engine.set_fen_position(board.fen())
    best_move_uci = engine.get_best_move_time(think_time_ms)
    
    if best_move_uci:
        return chess.Move.from_uci(best_move_uci)
    return None

def get_evaluation(engine, board):
    """Get Stockfish's evaluation of the position."""
    if engine is None:
        return 0
    
    engine.set_fen_position(board.fen())
    eval_dict = engine.get_evaluation()
    
    if eval_dict["type"] == "cp":
        return eval_dict["value"] / 100  # Convert centipawns to pawns
    elif eval_dict["type"] == "mate":
        mate_in = eval_dict["value"]
        return 999 if mate_in > 0 else -999
    return 0

# ==============================================================================
# UI LOGIC
# ==============================================================================

def render_board(board, last_move=None):
    """Render the chess board as SVG."""
    kwargs = {"board": board, "size": 400}
    if last_move:
        kwargs["lastmove"] = last_move
    
    board_svg = chess.svg.board(**kwargs)
    b64 = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
    st.write(f'<img src="data:image/svg+xml;base64,{b64}"/>', unsafe_allow_html=True)

st.set_page_config(page_title="Stockfish Chess", layout="centered")

st.title("♟️ Stockfish Chess")
st.write("Powered by Stockfish - the strongest open-source chess engine!")

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()

board = st.session_state.board

# Sidebar configuration
st.sidebar.subheader("⚙️ Engine Settings")
skill_level = st.sidebar.slider("Skill Level", 0, 20, 20, 
    help="0 = Beginner, 20 = Maximum strength")
think_time = st.sidebar.slider("Think Time (seconds)", 0.5, 5.0, 1.0, 0.5)
think_time_ms = int(think_time * 1000)

# Initialize engine with settings
engine = init_stockfish(depth=20, skill_level=skill_level)

if engine is None:
    st.error("⚠️ Stockfish not found! Please place `stockfish.exe` in the Chess folder.")
    st.info("Download from: https://stockfishchess.org/download/")
    st.stop()

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    last_move = board.move_stack[-1] if board.move_stack else None
    render_board(board, last_move)
    
    # Show evaluation
    eval_score = get_evaluation(engine, board)
    eval_bar = "⬜" * max(0, min(10, int(5 + eval_score)))
    eval_bar += "⬛" * (10 - len(eval_bar))
    
    if abs(eval_score) > 100:
        eval_text = f"Mate in {int(999 - abs(eval_score))}" if eval_score > 0 else f"Mate in {int(999 + eval_score)}"
    else:
        eval_text = f"{eval_score:+.2f}"
    
    st.write(f"**Evaluation:** {eval_text}")
    st.progress(min(1.0, max(0.0, (eval_score + 10) / 20)))

    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            st.success("🏆 White wins!")
        elif result == "0-1":
            st.success("🏆 Black wins!")
        else:
            st.info("🤝 Draw!")
        
        if st.button("🔄 New Game"):
            st.session_state.board = chess.Board()
            st.rerun()

with col2:
    st.subheader("🎮 Controls")
    
    # Human move input
    move_uci = st.text_input("Your Move (e.g., e2e4):", key="move_input")
    
    col_submit, col_engine = st.columns(2)
    
    with col_submit:
        if st.button("✅ Submit", use_container_width=True):
            if move_uci:
                try:
                    move = chess.Move.from_uci(move_uci.strip().lower())
                    if move in board.legal_moves:
                        board.push(move)
                        st.rerun()
                    else:
                        st.warning("Illegal move!")
                except ValueError:
                    st.warning("Invalid format! Use: e2e4")
    
    with col_engine:
        if st.button("🤖 Engine", use_container_width=True):
            if not board.is_game_over():
                with st.spinner(f"Thinking ({think_time}s)..."):
                    start = time.time()
                    best = get_best_move(engine, board, think_time_ms)
                    duration = time.time() - start
                    
                    if best:
                        board.push(best)
                        st.success(f"Played: {best.uci()} ({duration:.1f}s)")
                        st.rerun()
    
    st.divider()
    
    # Quick actions
    col_undo, col_new = st.columns(2)
    
    with col_undo:
        if st.button("↩️ Undo", use_container_width=True):
            if board.move_stack:
                board.pop()
                st.rerun()
    
    with col_new:
        if st.button("🆕 Reset", use_container_width=True):
            st.session_state.board = chess.Board()
            st.rerun()
    
    st.divider()
    
    # Show legal moves hint
    if st.checkbox("Show legal moves"):
        legal = [m.uci() for m in board.legal_moves]
        st.write(", ".join(legal[:20]) + ("..." if len(legal) > 20 else ""))

# Move history in sidebar
st.sidebar.divider()
st.sidebar.subheader("📜 Move History")
moves = list(board.move_stack)
for i in range(0, len(moves), 2):
    move_num = i // 2 + 1
    white_move = moves[i].uci()
    black_move = moves[i + 1].uci() if i + 1 < len(moves) else ""
    st.sidebar.write(f"{move_num}. {white_move} {black_move}")

# Engine info in sidebar
st.sidebar.divider()
st.sidebar.subheader("ℹ️ Engine Info")
st.sidebar.write(f"**Engine:** Stockfish")
st.sidebar.write(f"**Skill:** {skill_level}/20")
st.sidebar.write(f"**Think time:** {think_time}s")
