@echo off
REM MeshPrep v5 - Setup Script
REM Automates virtual environment creation and installation

echo.
echo ============================================================
echo MeshPrep v5 - Automated Setup
echo ============================================================
echo.

REM Check if venv already exists
if exist venv (
    echo Virtual environment already exists.
    choice /C YN /M "Do you want to recreate it"
    if errorlevel 2 goto activate
    echo.
    echo Removing old venv...
    rmdir /s /q venv
)

REM Try Python 3.12 first
echo Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    echo Found Python 3.12! Creating venv...
    py -3.12 -m venv venv
    goto install
)

REM Try Python 3.11
echo Python 3.12 not found. Checking for Python 3.11...
py -3.11 --version >nul 2>&1
if %errorlevel%==0 (
    echo Found Python 3.11! Creating venv...
    py -3.11 -m venv venv
    goto install
)

REM No compatible Python found
echo.
echo ERROR: Python 3.11 or 3.12 not found!
echo.
echo MeshPrep requires Python 3.11 or 3.12 (Open3D limitation)
echo.
echo Download from: https://www.python.org/downloads/
echo   - Python 3.12: https://www.python.org/downloads/release/python-3129/
echo   - Python 3.11: https://www.python.org/downloads/release/python-31111/
echo.
pause
exit /b 1

:install
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo Installing MeshPrep with all dependencies...
echo This may take 5-10 minutes (PyTorch is large)...
pip install -e ".[all]"

if %errorlevel%==0 (
    echo.
    echo ============================================================
    echo SUCCESS! MeshPrep v5 installed!
    echo ============================================================
    echo.
    echo To activate: venv\Scripts\activate
    echo To test: python test_runner_simple.py
    echo To use: meshprep repair model.stl
    echo.
) else (
    echo.
    echo ============================================================
    echo ERROR: Installation failed!
    echo ============================================================
    echo.
    echo See INSTALL.md for troubleshooting steps.
    echo.
)

pause
exit /b 0

:activate
echo.
echo To activate existing environment: venv\Scripts\activate
echo.
pause
