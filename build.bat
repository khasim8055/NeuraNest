@echo off
REM ================================================================
REM NeuraCare — Windows Build Script
REM ================================================================
REM Run this from the project root:
REM   build.bat
REM
REM Output: dist\NeuraCare\NeuraCare.exe
REM ================================================================

echo.
echo ================================================================
echo   NeuraCare Build System
echo ================================================================
echo.

REM Check we are in the right directory
if not exist "main.py" (
    echo ERROR: Run this script from the NeuraCare project root.
    echo        The folder containing main.py
    pause
    exit /b 1
)

REM Step 1: Activate virtual environment
echo [1/5] Activating virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: .venv not found, using system Python
)

REM Step 2: Install/upgrade build dependencies
echo [2/5] Installing build dependencies...
pip install pyinstaller --quiet
pip install pyinstaller-hooks-contrib --quiet

REM Step 3: Clean previous build
echo [3/5] Cleaning previous build...
if exist "dist\NeuraCare" rmdir /s /q "dist\NeuraCare"
if exist "build\NeuraCare" rmdir /s /q "build\NeuraCare"

REM Step 4: Build the exe
echo [4/5] Building NeuraCare.exe...
echo       This takes 2-5 minutes on first build.
echo.
pyinstaller neuracare.spec --noconfirm

REM Step 5: Copy required data files to dist
echo [5/5] Copying data files...
if not exist "dist\NeuraCare\app\data" mkdir "dist\NeuraCare\app\data"
copy "app\schema.sql" "dist\NeuraCare\app\schema.sql" >nul 2>&1
copy "demo_seed.py"   "dist\NeuraCare\demo_seed.py"   >nul 2>&1

REM Check result
if exist "dist\NeuraCare\NeuraCare.exe" (
    echo.
    echo ================================================================
    echo   BUILD SUCCESSFUL
    echo   Output: dist\NeuraCare\NeuraCare.exe
    echo.
    echo   To distribute:
    echo   1. Zip the entire dist\NeuraCare\ folder
    echo   2. Send to clinic
    echo   3. They unzip and run NeuraCare.exe
    echo ================================================================
) else (
    echo.
    echo ================================================================
    echo   BUILD FAILED
    echo   Check the error messages above.
    echo   Common fixes:
    echo   - Run: pip install pyinstaller
    echo   - Run: pip install PyQt6 reportlab bcrypt matplotlib
    echo ================================================================
)

echo.
pause
