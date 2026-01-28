
import ctypes
import os
import time

# Load the DLL directly
dll_path = os.path.join(os.path.dirname(__file__), "target", "release", "rust_engine.dll")
if not os.path.exists(dll_path):
    print(f"Error: DLL not found at {dll_path}")
    exit(1)

lib = ctypes.CDLL(dll_path)

# Define Argument Types
lib.get_best_move_c.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.get_best_move_c.restype = ctypes.c_char_p # Returns pointer to string

lib.free_string.argtypes = [ctypes.c_char_p]
lib.free_string.restype = None

lib.evaluate_c.argtypes = [ctypes.c_char_p]
lib.evaluate_c.restype = ctypes.c_int

FEN = "5QQ1/8/8/8/2P1k3/8/5R2/3rK3 w - - 1 81"

def main():
    print(f"Debugging Stuck Position (via ctypes DLL)...")
    print(f"FEN: {FEN}")
    
    fen_bytes = FEN.encode('utf-8')
    
    start = time.time()
    print("Getting best move (Depth 10)...")
    try:
        # Call C function: get_best_move_c(fen, depth, threads)
        # Using 1 thread for serial debugging
        ptr = lib.get_best_move_c(fen_bytes, 10, 1)
        
        if ptr:
            move_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')
            print(f"Best Move: {move_str}")
            lib.free_string(ptr)
        else:
            print("Error: get_best_move_c returned NULL")
            
    except Exception as e:
        print(f"CRASH: {e}")
        
    end = time.time()
    print(f"Time: {end - start:.3f}s")
    
    print("Evaluating position...")
    try:
        score = lib.evaluate_c(fen_bytes)
        print(f"Score: {score}")
    except Exception as e:
        print(f"Eval Crash: {e}")

if __name__ == "__main__":
    main()
