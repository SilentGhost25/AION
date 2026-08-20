#!/usr/bin/env bash
set -e
cd /home/AIML1/AIQ/AION

pkill -f aion_api.py 2>/dev/null || true
pkill -f aion_launcher.py 2>/dev/null || true
sleep 1

export AION_PROFILE="PRODUCTION"
export AION_MODEL="qwen2.5:14b"
export PYTHONPATH="/home/AIML1/AIQ/AION/patches:/home/AIML1/AIQ/AION:${PYTHONPATH}"

exec python3 /home/AIML1/AIQ/AION/patches/aion_launcher.py
