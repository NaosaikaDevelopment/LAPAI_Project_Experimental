import os
import re
import sys
import json
import time
import faiss
import sqlite3
import threading
import itertools
import subprocess
import numpy as np
from pathlib import Path
import onnxruntime as ort
from openai import OpenAI
from datetime import datetime, timezone
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
def _abs(p, base):
    p = Path(p)
    return p if p.is_absolute() else base / p
def load_config(file_path=None):
    if file_path is None:
        file_path = ROOT / "Settings" / "conf.json"
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Using defaults.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: {file_path} contains invalid JSON.")
        return {}


class LAPAICache:
    def __init__(self):
        self.conf = load_config()
        self.EMBED_DIM = self.conf.get('EMBEDDIM')
        self.DATA_DIR = _abs(os.environ.get("LAPAI_DATA") or self.conf.get("datadir") or ".", ROOT)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.persnoal_file = str(self.DATA_DIR / "PersonalData.txt")
        self.CHAT_DIR = str(self.DATA_DIR / "chats")
        self.DB_FILE = str(self.DATA_DIR / "Raw_Memory.db")
        self.Learning_Dir = str(self.DATA_DIR / "Learning")
        self.FAISS_INDEXM = str(self.DATA_DIR / "Main_Memory.bin")
        self.FAISS_MAP = str(self.DATA_DIR / "memory_map.json")
        self.QUESTIONS_FILE = str(self.DATA_DIR / "questions.json")

        self.pepath = str(self.conf.get('personapath'))
        self.OLLAMA_BASE_URL = self.conf.get('ollamaurl', "http://localhost:11434/v1")
        self.BASE_URL = self.conf.get('baseurl')
        self.model_name = self.conf.get('mainmodel')
        self.Sum_model = self.conf.get('summodel')
        self.API_KEY_DM = "dummy"

        self.model_path_embed = _abs(self.conf.get('embedmodelpath'), ROOT)
        self.Learning_DB = os.path.join(self.Learning_Dir, "Learning.db")
        self.tokenizer = AutoTokenizer.from_pretrained(str(_abs(self.conf.get('embedtokenizer'), ROOT)))
        self.client = OpenAI(base_url=self.BASE_URL, api_key=self.API_KEY_DM)
        self.session = ort.InferenceSession(str(self.model_path_embed), providers=[self.conf.get('provider')])

        self.id_map = {}
        self.faiss_index = None
        self.seid = {}
        self.jsfile = None
        self.session_id = {}
        self.session_file = None
        self.prompt = []


cache = LAPAICache()
