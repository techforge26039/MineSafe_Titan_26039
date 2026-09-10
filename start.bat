@echo off
cd /d "%~dp0"

echo ==========================================
echo MineSafe - SIH 2026
echo ==========================================
echo.

python -m pip install -r requirements.txt

echo.
echo Starting MineSafe Streamlit dashboard...
echo.

python -m streamlit run minesafe/app.py

pause
