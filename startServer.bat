@echo off
cd "SEU_CAMINHO\ServerConfig"

:: Ativa a venv
call .\venv\Scripts\activate

:: Inicia o servidor Python em uma nova janela
start cmd /k "python server.py"

:: Espera 5 segundos para garantir que o servidor iniciou
timeout /t 5 >nul

:: Inicia o ngrok em outra janela
start cmd /k "ngrok http 5000"