#!/bin/sh

python3.10 -m venv LAPAI-env

LAPAI-env/bin/python3.10 -m pip install --upgrade pip
LAPAI-env/bin/python3.10 -m pip install -r requirements-l.txt
git lfs install
cd MainCore
git clone https://huggingface.co/intfloat/multilingual-e5-small

echo
echo "If model downloaded just <10mb than you need install git-lfs"
echo
echo "installation done..."

read -p "Press Enter to continue..."
