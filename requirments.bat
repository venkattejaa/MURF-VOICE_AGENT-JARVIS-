@echo off
title J.A.R.V.I.S AI - Setup Installer
color 0A

echo ======================================================
echo        J.A.R.V.I.S AI (BUILT BY MURF FALCON-TTS)
echo                    AUTO INSTALLER
echo        Created by: S. Venkata Teja Naik (DCME)
echo ======================================================
echo.

:: ---------- CHECK PYTHON ----------
echo Checking Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Python is not installed.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    pause
    exit /b
)

echo Python detected.
echo.

:: ---------- INSTALL DEPENDENCIES ----------
echo Installing required Python modules...
echo This may take a moment...
echo.

pip install --upgrade pip >nul

pip install deepgram-sdk >nul
pip install murf >nul
pip install groq >nul
pip install websockets >nul
pip install sounddevice >nul
pip install numpy >nul
pip install pywebview >nul
pip install pywhatkit >nul
pip install python-dotenv >nul
pip install requests >nul

echo.
echo Dependencies installed successfully.
echo.

:: ---------- CREATE .ENV FILE ----------
echo Setting up configuration...

IF exist ".env" (
    echo .env already exists. Skipping creation.
) ELSE (
    echo Creating .env file...
    (
        echo MURF_API_KEY=ap2_085ec4a2-eaf7-4d80-a6e7-70d3bcabfb28
        echo DEEPGRAM_API_KEY=988be47d91aca476e11aad90ed37e5abf4d34eb9 
        echo GROQ_API_KEY=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
        echo CREATOR_NAME=S.Venkata Teja
        echo WS_PORT=8765
        echo WAKEWORD=jarvis
        echo AI_MODEL=llama-3.3-70b-versatile
        echo TTS_BOOST=12
    ) > .env
    echo .env file created. Please edit it with your keys.
)

echo.



echo.
echo J.A.R.V.I.S session ended.
pause



