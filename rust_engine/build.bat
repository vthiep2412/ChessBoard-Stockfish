@echo off
REM Get the directory where this script is located
cd /d "%~dp0"

echo Building Rust Engine...
cargo build --release
if %errorlevel% neq 0 exit /b %errorlevel%

echo Copying DLL to PYD...
copy /Y "target\release\rust_engine.dll" "target\release\rust_engine.pyd"
echo Build Complete!
