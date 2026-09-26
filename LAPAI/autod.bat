@echo off
cd /d "%~dp0"
python3.10 -m venv LAPAI-env
LAPAI-env\Scripts\python -m pip install --upgrade pip
LAPAI-env\Scripts\python -m pip install -r requirements-w.txt
LAPAI-env\Scripts\python -m pip install -r requirements-strict.txt
cd MainCore
git clone https://huggingface.co/intfloat/multilingual-e5-small
pip install pyside6

setlocal enabledelayedexpansion

set "VENV_PATH=%~dp0LAPAI-env\Scripts\activate.bat"


set "SHORTCUT_DIR=%USERPROFILE%\AppData\Local\Microsoft\WindowsApps"
set "SHORTCUT_FILE=%SHORTCUT_DIR%\nd.bat"

echo @echo off > "%SHORTCUT_FILE%"
echo if exist "%VENV_PATH%" ( >> "%SHORTCUT_FILE%"
echo     call "%VENV_PATH%" >> "%SHORTCUT_FILE%"
echo     echo ^[✓^] Environment activated! >> "%SHORTCUT_FILE%"
echo ) else ( >> "%SHORTCUT_FILE%"
echo     echo ^[✗^] File venv not found: %VENV_PATH% >> "%SHORTCUT_FILE%"
echo ) >> "%SHORTCUT_FILE%"

echo [✓] 'nd' Succeded add to global Windows!
echo quick explanation, to use this project env you can now easily use 'nd' in console to turn on the Environment
cd ../
./LAPAI-env/bin/python toreg.py
echo installation done...
pause
