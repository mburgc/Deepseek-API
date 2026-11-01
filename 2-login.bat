@echo off
setlocal
title DeepSeek Bridge - Login
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] Primero ejecuta 1-setup.bat
  pause
  exit /b 1
)

call venv\Scripts\activate.bat

echo ==========================================
echo  Login de DeepSeek
echo ==========================================
echo.
echo Se va a abrir una ventana de navegador.
echo.
echo   1. Inicia sesion con TU cuenta de chat.deepseek.com
echo   2. Resuelve el "verify you are human" si aparece
echo   3. Cuando veas el chat, cerra la ventana
echo.
echo La sesion queda guardada en la carpeta session\
echo (no la compartas con nadie: son tus cookies).
echo.
pause

python -m deepseek.auth

echo.
if exist "session\session.json" (
  echo [OK] Sesion guardada.
  echo      Ahora podes ejecutar 3-run.bat
) else (
  echo [AVISO] No se encontro session\session.json
  echo         Proba de nuevo o revisa el LEEME.md
)
pause
