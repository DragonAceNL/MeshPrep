@echo off
echo Starting MeshPrep Dashboard Server...
echo.
echo Dashboard will be available at: http://localhost:8080/live_dashboard.html
echo Press Ctrl+C to stop the server.
echo.
start http://localhost:8080/live_dashboard.html
cd /d "C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v3"
python -m http.server 8080
