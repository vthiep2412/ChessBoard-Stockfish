@echo off
setlocal enabledelayedexpansion
REM Stockfish Quick Test Script - 3 runs with average
REM Compares baseline vs modified builds - shows NPS and TIME

REM Change to src folder
cd /d "%~dp0src"

echo.
echo ============================================================
echo   STOCKFISH QUICK BENCHMARK (3 runs each)
echo   Showing: Nodes/second and Time
echo ============================================================

REM Check if baseline exists
if not exist "stockfish_baseline.exe" (
    echo.
    echo [!] stockfish_baseline.exe not found!
    goto :setup
)

REM Check if modified exists  
if not exist "stockfish.exe" (
    echo.
    echo [!] stockfish.exe not found!
    goto :end
)

REM BASELINE TESTS
echo.
echo ================================================
echo   TESTING BASELINE (3 runs)
echo ================================================
set /a baseline_total=0

for /L %%i in (1,1,3) do (
    echo   Run %%i of 3...
    for /f "tokens=*" %%a in ('stockfish_baseline.exe bench 2^>^&1 ^| findstr /C:"Nodes/second" /C:"Total time"') do (
        echo     %%a
    )
    for /f "tokens=3" %%a in ('stockfish_baseline.exe bench 2^>^&1 ^| findstr /C:"Nodes/second"') do (
        set /a baseline_total+=%%a
    )
)
set /a baseline_avg=baseline_total/3
echo   ---------------------------------
echo   BASELINE AVERAGE: !baseline_avg! NPS

REM MODIFIED TESTS
echo.
echo ================================================
echo   TESTING MODIFIED (3 runs)
echo ================================================
set /a modified_total=0

for /L %%i in (1,1,3) do (
    echo   Run %%i of 3...
    for /f "tokens=*" %%a in ('stockfish.exe bench 2^>^&1 ^| findstr /C:"Nodes/second" /C:"Total time"') do (
        echo     %%a
    )
    for /f "tokens=3" %%a in ('stockfish.exe bench 2^>^&1 ^| findstr /C:"Nodes/second"') do (
        set /a modified_total+=%%a
    )
)
set /a modified_avg=modified_total/3
echo   ---------------------------------
echo   MODIFIED AVERAGE: !modified_avg! NPS

REM COMPARISON
echo.
echo ============================================================
echo   COMPARISON RESULTS
echo ============================================================
set /a diff=modified_avg-baseline_avg
set /a pct_x100=diff*10000/baseline_avg

echo   Baseline Average: !baseline_avg! NPS
echo   Modified Average: !modified_avg! NPS
echo   Difference:       !diff! NPS

if !diff! GTR 0 (
    echo.
    echo   *** MODIFIED IS FASTER! ***
) else if !diff! LSS 0 (
    echo.
    echo   *** MODIFIED IS SLOWER ***
) else (
    echo.
    echo   *** NO DIFFERENCE ***
)
echo ============================================================
echo.
goto :end

:setup
echo.
echo SETUP: Copy stockfish.exe to stockfish_baseline.exe first
echo.

:end
endlocal
pause
