# Chess Web App with Stockfish

A Streamlit-powered chess application using the Stockfish engine.

## Features
- 🎮 Play against Stockfish
- ⚡ Adjustable skill level (0-20)
- ⏱️ Configurable think time
- 📊 Real-time position evaluation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Stockfish Engine

This project includes the Stockfish source code (GPL-3.0 licensed).
The engine binary should be placed in the root folder as `stockfish.exe`.

To build from source (requires MinGW/MSYS2):
```bash
cd Stockfish-master/src
make -j build ARCH=x86-64
```

## License

- Web app code: MIT License
- Stockfish engine: GPL-3.0 (see Stockfish-master/Copying.txt)
