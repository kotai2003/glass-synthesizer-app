@echo off
REM Launch the GLASS Synthesizer GUI with the glass_env Python.
REM Run by double-click or from cmd:
REM     run.bat
setlocal
set GLASS_PY=C:\Users\seong\anaconda3\envs\glass_env\python.exe
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%\.."
"%GLASS_PY%" -m synthesizer_app.gui_main %*
endlocal
