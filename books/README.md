# Opening Books for AntigravityRust

## How to Get Opening Books

Download any of these Polyglot (.bin) books and place in this folder:

### Recommended Books:

1. **Cerebellum** (Strongest)
   - Download: https://github.com/nicvagn/cerebellum_light/releases
   - Rename to: `cerebellum.bin`

2. **Perfect2023** (Very Deep)
   - Download: https://www.mediafire.com/folder/o9g9v6bxeuo2x/ChessBooks
   - Look for: `Perfect2023.bin`

3. **Titans** (Good Balance)
   - Download: https://github.com/ChessDB/Polyglot-books/tree/master/Performance
   - Rename to: `titans.bin`

## File Naming

Place files with these exact names:
```
books/
├── cerebellum.bin    (priority 1)
├── perfect2023.bin   (priority 2)
├── titans.bin        (priority 3)
└── opening.bin       (fallback)
```

The engine will use the first book it finds!

## Quick Test

After adding a book, test with:
```python
import rust_engine
print(rust_engine.has_book_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
# Should print: True
```
