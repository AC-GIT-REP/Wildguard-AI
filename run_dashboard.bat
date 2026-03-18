@echo off
cd /d "%~dp0"
echo ===================================================
echo   WildGuard AI - Endangered Species Dashboard
echo ===================================================
echo.
echo Starting Streamlit Server...
echo The dashboard will open in your default browser.
echo.
streamlit run app/app.py
pause
