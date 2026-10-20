#!/usr/bin/env bash
# SecurePay: one-step setup and launch for Linux and macOS.
# First run creates a virtual environment and installs the libraries (a few minutes).
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment ..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  if [ "$(uname)" = "Linux" ]; then
    # CPU build of PyTorch is ~200 MB instead of ~2 GB
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

streamlit run app.py
