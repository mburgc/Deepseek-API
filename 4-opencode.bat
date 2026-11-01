@echo off
setlocal
title DeepSeek Bridge - Configurar opencode
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] Primero ejecuta 1-setup.bat
  pause
  exit /b 1
)

call venv\Scripts\activate.bat

echo ==========================================
echo  Configurar opencode
echo ==========================================
echo.
echo Se agrega el proveedor "deepseek-api" a tu config de opencode.
echo No borra nada: hace backup y fusiona.
echo.

python setup_opencode.py

echo.
echo Si tu config tiene comentarios, el script no la toca:
echo en ese caso pega el bloque de opencode.jsonc a mano.
echo.
pause
