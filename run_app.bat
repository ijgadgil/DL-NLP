@echo off
REM SecurePay: one-step setup and launch for Windows.
REM First run creates a virtual environment and installs the libraries (a few minutes).
cd /d "%~dp0"

if not exist ".venv" (
  echo Creating virtual environment ...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)

streamlit run app.py
