#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import sys; assert sys.version_info >= (3,11), "Use Python 3.11+, preferably 3.12"; print(sys.version)'
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
