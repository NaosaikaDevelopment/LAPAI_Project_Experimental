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
class LAPAICache:
    def __init__(self):
        self.EMBED_DIM = 384 
        self.CHAT_DIR = "chats"
        self.DB_FILE = "Raw_Memory.db"
        self.Learning_Dir = "Learning"
        self.FAISS_INDEXM = "Main_Memory.bin"
        self.FAISS_MAP = "memory_map.json"
        self.QUESTIONS_FILE = "questions.json"
        self.OLLAMA_BASE_URL = "http://localhost:11434/v1"

        #
        if os.path.exists("Settings/baseurl"):
            with open("Settings/baseurl", "r", encoding="utf-8") as f:
                self.BASE_URL = f.read().strip()
        #MAIN MODEL
        if os.path.exists("Settings/1MainNameModel.txt"):
            with open("Settings/1MainNameModel.txt", "r", encoding="utf-8") as f:
                self.model_name = f.read().strip()

        #Sum_model
        if os.path.exists("Settings/1SumNameModel.txt"):
            with open("Settings/1SumNameModel.txt", "r", encoding="utf-8") as f:
                self.Sum_model = f.read().strip()

        LEMONADE_API_KEY = "lemonade"
        self.model_path_embed = Path("MainCore/multilingual-e5-small/onnx/model.onnx").resolve()
        self.Learning_DB = os.path.join("Learning", "Learning.db")
        self.tokenizer = AutoTokenizer.from_pretrained(Path("MainCore/multilingual-e5-small").resolve())
        self.client = OpenAI(base_url=self.BASE_URL, api_key=LEMONADE_API_KEY)
        self.session = ort.InferenceSession(str(self.model_path_embed), providers=["CPUExecutionProvider"])
        
        self.id_map = {}
        self.faiss_index = None
        self.seid = {}
        self.jsfile = None
        self.session_id={}
        self.session_file=None
        self.prompt =[]


cache = LAPAICache()