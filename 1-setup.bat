@echo off
setlocal
title DeepSeek Bridge - Instalacion
cd /d "%~dp0"

echo ==========================================
echo  DeepSeek Bridge - Instalacion
echo  (se ejecuta una sola vez)
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python no esta en el PATH.
  echo.
  echo Instalalo desde https://www.python.org/downloads/
  echo y marca la casilla "Add python.exe to PATH" al inicio.
  echo.
  pause
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)"
if errorlevel 1 (
  echo [ERROR] Hace falta Python 3.9 o superior.
  python --version
  pause
  exit /b 1
)

echo [1/4] Creando entorno virtual...
if not exist "venv\Scripts\python.exe" (
  python -m venv venv
  if errorlevel 1 goto :fail
) else (
  echo       ya existe, se reutiliza
)

echo [2/4] Instalando dependencias...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [3/4] Descargando Chromium para Playwright...
echo       (la primera vez son ~150 MB)
python -m playwright install chromium
if errorlevel 1 goto :fail

echo [4/4] Diagnostico...
echo.
python doctor.py

echo.
echo ==========================================
echo  Siguiente paso: ejecutar 2-login.bat
echo ==========================================
pause
exit /b 0

:fail
echo.
echo [ERROR] La instalacion fallo. Copia el mensaje de arriba
echo         y revisa el LEEME.md, seccion Problemas.
pause
exit /b 1
