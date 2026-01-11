@echo off
REM Deep analysis benchmark at depth 14
cd /d "%~dp0"

echo Building...
cargo build --release 2>nul

echo.
echo Running deep benchmark (depth 14 - may take a few minutes)...
python benchmark.py 14

pause
