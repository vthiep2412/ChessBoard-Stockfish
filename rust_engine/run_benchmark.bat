@echo off
REM ============================================
REM Chess Engine Testing Suite
REM ============================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================
echo    Chess Engine Benchmark Suite
echo ============================================
echo.

REM Check if we need to build
if "%1"=="--skip-build" goto :run_tests

echo [1/3] Building release version...
cargo build --release
if errorlevel 1 (
    echo ERROR: Build failed!
    exit /b 1
)
echo Build successful!
echo.

:run_tests
echo [2/3] Running Python benchmark...
python benchmark.py %*

echo.
echo [3/3] Done!
pause
