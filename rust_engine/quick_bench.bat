@echo off
REM Quick NPS test at depth 12
cd /d "%~dp0"

echo Building...
cargo build --release 2>nul

echo.
echo Running quick benchmark (depth 12)...
python benchmark.py 12

pause
