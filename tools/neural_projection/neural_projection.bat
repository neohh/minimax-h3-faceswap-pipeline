@echo off
setlocal enabledelayedexpansion

:: CoTracker3 + Smooth-MLS Neural Texture Projection Runner
:: Part of VFX Studio Trifecta: Block 3

set "PY="
if exist "Z:\Work\VFX_Program\Sammie-Roto-2-main\.venv\Scripts\python.exe" (
    set "PY=Z:\Work\VFX_Program\Sammie-Roto-2-main\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

if "%~1"=="" (
    echo [!] Usage: neural_projection.bat --video PLATE.mp4 --mask MASK.mp4 --hero HERO.png --output OUT.mp4
    echo.
    "%PY%" "%~dp0neural_projection.py" --help
    exit /b 1
)

"%PY%" "%~dp0neural_projection.py" %*
