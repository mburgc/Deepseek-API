@echo off
setlocal
title DeepSeek Bridge - Servidor
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] Primero ejecuta 1-setup.bat
  pause
  exit /b 1
)
if not exist "session\session.json" (
  echo [ERROR] No hay sesion. Ejecuta 2-login.bat primero.
  pause
  exit /b 1
)

call venv\Scripts\activate.bat

echo ==========================================
echo  DeepSeek Bridge
echo  http://127.0.0.1:8000/v1
echo ==========================================
echo.
echo Deja esta ventana abierta mientras lo uses.
echo Ctrl+C para detener el servidor.
echo.

python app.py

echo.
echo El servidor se detuvo.
pause
