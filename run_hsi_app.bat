@echo off
REM ============================================================================
REM  HSI Control App Launcher
REM  This script activates the virtual environment and runs the main Python app.
REM ============================================================================

REM --- Change directory to the script's own location ---
REM This makes all subsequent relative paths reliable, no matter where
REM this batch file is called from. %~dp0 is the drive and path of the script.
pushd "%~dp0"
echo Navigated to: %cd%

REM --- Activate the virtual environment ---
REM The path '..\..\venv' points two directories up from this script's location.
echo Activating virtual environment...
call build_venv\Scripts\activate.bat

REM --- Error Checking: Verify that the venv was activated ---
REM The 'activate.bat' script sets a variable called VIRTUAL_ENV.
REM If it's not set, the path was wrong or the venv is broken.
if "%VIRTUAL_ENV%"=="" (
    echo.
    echo ERROR: Could not activate the virtual environment.
    echo Please check that the path is correct.
    echo.
    pause
    exit /b 1
)

REM --- Run the main Python application ---
echo Launching HSI Control App...
"build_venv\Scripts\python.exe" hsi_control_v5.py

REM --- Cleanup ---
REM Return to the original directory if it was changed by pushd
popd

echo Application finished.
REM Uncomment the next line for debugging if the window closes too quickly.
REM pause