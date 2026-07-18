@echo off
echo === Unhuman Clips Backend ===
echo Starting server on http://127.0.0.1:8000
echo.

REM Note:
REM - This .bat is for Windows terminals.
REM - In WSL, prefer: ./start.sh
REM - We keep it working by launching uvicorn directly from the backend folder
REM   and using the local venv if present.

set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%backend"

REM Try venv Python first, fall back to system Python
if exist "%BACKEND_DIR%\venv\Scripts\python.exe" (
  "%BACKEND_DIR%\venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 --app-dir "%PROJECT_DIR%"
) else (
  echo Warning: venv not found, using system Python
  python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 --app-dir "%PROJECT_DIR%"
)

pause
