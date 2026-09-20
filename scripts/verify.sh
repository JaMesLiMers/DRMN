#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/check_sources.py
