@echo off
echo ========================================
echo   DevRAG Backend Development Shell
echo ========================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.
echo Available commands:
echo   python run.py           - Start backend server
echo   python tests\benchmark_suite.py - Run benchmarks
echo   pip install -r requirements.txt - Update dependencies
echo.
cmd /k
