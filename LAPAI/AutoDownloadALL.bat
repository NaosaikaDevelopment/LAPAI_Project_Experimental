
python3.10 -m venv LAPAI-env

LAPAI-env\Scripts\python -m pip install --upgrade pip
LAPAI-env\Scripts\python -m pip install -r requirements-w.txt
cd MainCore
git clone https://huggingface.co/intfloat/multilingual-e5-small

echo installation done...
pause
