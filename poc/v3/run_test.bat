@echo off
echo ============================================================
echo MeshPrep Thingi10K Full Test
echo ============================================================
echo.
echo Starting processing... Output will appear below.
echo Press Ctrl+C to stop.
echo.

cd /d "C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v3"
set PYTHONUNBUFFERED=1
..\v2\.venv312\Scripts\python.exe -u run_full_test.py %*

echo.
echo Script finished.
pause
