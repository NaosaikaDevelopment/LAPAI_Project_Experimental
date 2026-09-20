@echo off
set "SHORTCUT_FILE=%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\nd.bat"
if exist "%SHORTCUT_FILE%" (
    del "%SHORTCUT_FILE%"
    echo [✓] The 'nd' shortcut has been removed from your Windows system!
) else (
    echo [ℹ] The 'nd' shortcut was not found or has already been uninstalled.
)
pause
