import os
import re
import json
import time
import socket
import sqlite3
import requests
import subprocess
from pathlib import Path
from openai import OpenAI
from datetime import datetime












"""
ATTENTION:
This is Based From Lemonade server and make sure you had two model, in Lemonade Itself im using CPU model NPU model, and there available model using NPU/GPU/CP/Hybrid so be wise 
check on their website if you want learn more

what is sumerize model?
Ah dont worry its just regular light weight model and the function just make conclusion and question
you can use another model, the model you use maybe using GPU/CPU?
its Fine.

your model using NPU?
its fine too. (In case here NPU im using Lemonade Server, but if you make or found another Program that using API NPU model, its fine too)

Have Fun! and Goodluck!
"""


















LEMONADE_BASE_URL = "http://localhost:8000/api/v1" #<--- API AI, Up to you for choosing
LEMONADE_API_KEY = "lemonade"

MODEL_NAME = "Phi-3.5-Mini-Instruct-NPU" #<--- Attention here, if you wanna change model learning at here (Using Lemonade server)
SUM_MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B-NPU"#<---this is model for sumerize, be sure that model is light weight if you wanna change
#Learn and load from history
CHAT_DIR = "chats"
DB_FILE = "memory.db"
#Learn import
Learning_Dir = "Learning"
DB_FILEL = "Learn.db"
#online mode [Experimental] 
API_KEY = "FILL_YOUR_API_KEY"   # Google Custom Search API Key
CX = "FILL_CUSTOM_SEARCH_ID"    # Google Programmable Search Engine ID
"""
Note:
this feature is work for dynamic intellegence so your AI can be more personal
but keep in mind this is still experimental, if you won't using it just keep it like that

Check Online Tutorial How to get Google API, and goodluck
"""
QUESTIONS_FILE = "questions.json"
#Space Time
LEARNING_INTERVAL = 60 * 30 
os.makedirs(Learning_Dir, exist_ok=True)


#----Utility and function for online mode-----
def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def research_questions():
    questions = load_json(QUESTIONS_FILE, [])
    updated = False

    for q in questions:
        if not q["answered"]:
            print(f"[SEARCH] {q['question']}")
            results = google_search(q["question"])
            if results:
                q["answer"] = results
                q["answered"] = True
                updated = True

    if updated:
        save_json(QUESTIONS_FILE, questions)


def google_search(query, num=3): #<---and keep watch on here if some bug like error handling
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": CX, "q": query, "num": num}
    res = requests.get(url, params=params)
    
    if res.status_code != 200:
        return {"error": f"Request failed: {res.status_code}"}
    data = res.json()

    results = []
    if "items" in data:
        for item in data["items"]:
            title = item.get("title")
            snippet = item.get("snippet")
            link = item.get("link")
            results.append({
                "title": item["title"],
                "link": item["link"],
                "snippet": item["snippet"]
            })
    return results


def extract_intel_from_research(client, model, session_id, session_file, question):
    search_results = google_search(question)

    if not search_results or (isinstance(search_results, dict) and "error" in search_results):
        append_Learning(session_id, session_file, "research", f"[NO RESULTS] {question}")
        return f"[NO RESULTS] {question}"

    if not isinstance(search_results, list):
        search_results = [search_results]

    combined_text = "\n".join(
        [f"{r.get('title','')} - {r.get('snippet','')} ({r.get('link','')})" for r in search_results]
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a research summarizer. Extract key facts only."},
            {"role": "user", "content": combined_text}
        ],
        max_tokens=400
    )

    intel = completion.choices[0].message.content.strip()
    append_Learning(session_id, session_file, "research", intel)
    return intel


#Function Area:
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        json_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(
        session_id UNINDEXED,
        role,
        content
    )
    """)

    conn.commit()   
    conn.close()


def init_db_learning():
    """Buat tabel-tabel learning termasuk tabel questions."""
    os.makedirs(Learning_Dir, exist_ok=True)
    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()

    # sessions table (with learned flag)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        json_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        learned INTEGER DEFAULT 0
    )
    """)

    # full-text messages (FTS5)
    c.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(
        session_id UNINDEXED,
        role,
        content
    )
    """)

    # questions table (for migrating questions.json -> DB)
    c.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT UNIQUE,
        answered INTEGER DEFAULT 0,
        answer TEXT,
        timestamp TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def load_json(path, default=[]):
    if not os.path.exists(path):
        return [] if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


#memory management Learning
def create_session(title):
    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()

    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", title) + ".json"
    json_file = os.path.join(Learning_Dir, filename)

    c.execute("INSERT INTO sessions (title, json_file) VALUES (?, ?)", (title, json_file))
    conn.commit()
    sid = c.lastrowid
    conn.close()

    if not os.path.exists(json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    return sid, json_file



def append_Learning(session_id, json_file, role, content):
    history = []
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append({"role": role, "content": content})
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()



def search_memory(query, limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    SELECT session_id, role, content 
    FROM messages 
    WHERE messages MATCH ? 
    LIMIT ?
    """, (query, limit))
    rows = c.fetchall()
    conn.close()
    return rows

def load_history(json_file):
    if not os.path.exists(json_file):
        return []
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)




#Learning Function
def upgrade_learning_db():
    conn = sqlite3.connect(DB_FILEL)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN learned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  
    conn.commit()
    conn.close()

def upgrade_memory_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN learned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  
    conn.commit()
    conn.close()

def ambil_sesi_belum_dipelajari():
    conn = sqlite3.connect(DB_FILEL)
    cur = conn.cursor()
    cur.execute("SELECT id, json_file FROM sessions WHERE learned = 0 LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0], row[1]  #(session_id, session_file)
    return None, None
    
def tandai_sudah_dipelajari(session_id):
    conn = sqlite3.connect(DB_FILEL)
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET learned = 1 WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def generate_question(client, model, text):
    prompt = f"You are detective AI, Your job is search question and behavior of user from this text: {text}\n\n"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()
def add_question(question):
    questions = load_json(QUESTIONS_FILE, [])
    if any(q["question"] == question for q in questions):
       print("[SKIP] Question already exists.")
       return
    questions.append({
        "question": question,
        "answered": False,
        "timestamp": datetime.now().isoformat(),
        "answer": []
    })
    save_json(QUESTIONS_FILE, questions)
def import_questions_json_to_db():
    qs = load_json(QUESTIONS_FILE, [])
    if not qs:
        return 0, 0

    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()
    inserted, updated = 0, 0
    now = datetime.now().isoformat()

    for item in qs:
        qtext = (item.get("question") or "").strip()
        if not qtext:
            continue
        answered = 1 if item.get("answered") else 0
        answer = item.get("answer")
        if answer is not None and not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)

        c.execute("SELECT id, answered FROM questions WHERE question = ?", (qtext,))
        row = c.fetchone()
        if row:
            qid, old_ans = row
            if answered and not old_ans:
                c.execute(
                    "UPDATE questions SET answered=?, answer=?, WHERE id=?",
                    (answered, answer, qid)
                )
                updated += 1
        else:
            c.execute(
                "INSERT INTO questions(question, answered, answer) VALUES (?,?,?)",
                (qtext, answered, now)
            )
            inserted += 1

    conn.commit()
    conn.close()
    return inserted, updated

def add_question_db(question: str):
    if not question or not question.strip():
        return
    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO questions(question) VALUES (?)", (question.strip(),))
        conn.commit()
        print("[OK] Question inserted to DB.")
    except sqlite3.IntegrityError:
        print("[SKIP] Question already exists in DB.")
    finally:
        conn.close()


def migrate_questions_json_to_db():
    """Baca questions.json dan masukkan ke tabel questions bila belum ada."""
    questions = load_json(QUESTIONS_FILE, [])
    if not questions:
        return

    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()

    for q in questions:
        qtext = q.get("question")
        if not qtext:
            continue
        # cek sudah ada
        c.execute("SELECT id FROM questions WHERE question = ? LIMIT 1", (qtext,))
        if c.fetchone():
            continue
        answered = 1 if q.get("answered") else 0
        answer_text = json.dumps(q.get("answer")) if isinstance(q.get("answer"), (list, dict)) else (q.get("answer") or "")
        ts = q.get("timestamp") or datetime.now().isoformat()
        c.execute("INSERT INTO questions (question, answered, answer, timestamp) VALUES (?, ?, ?, ?)",
                  (qtext, answered, answer_text, ts))

    conn.commit()
    conn.close()
def fetch_unanswered_questions(limit=10):
    """Ambil pertanyaan yang belum dijawab dari DB."""
    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()
    c.execute("SELECT id, question FROM questions WHERE answered = 0 ORDER BY id ASC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
def append_question_to_db(question_text):
    """Tambahkan pertanyaan baru ke DB (jika belum ada)."""
    conn = sqlite3.connect(DB_FILEL)
    c = conn.cursor()
    c.execute("SELECT 1 FROM questions WHERE question = ? LIMIT 1", (question_text,))
    if c.fetchone():
        conn.close()
        return False
    c.execute("INSERT INTO questions (question, answered, answer, timestamp) VALUES (?, 0, '', ?)",
              (question_text, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return True


#Sum Info
def summarize_session(client, model, session_id, session_file, max_tokens=500):
    """
    Create a brief summary of a long topic. 
    The summary result is stored in the database & JSON in learning.
    """
    history = load_history(session_file)
    if len(history) < 15:
        return None

    text_blocks = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        text_blocks.append(f"{role.upper()}: {content}")

    joined_text = "\n".join(text_blocks)

    if len(joined_text.split()) > 1000:
        joined_text = " ".join(joined_text.split()[-1000:])

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a summarizer AI. Summarize this Topic in a concise way, keep facts, remove filler."},
            {"role": "user", "content": joined_text}
        ],
        max_tokens=max_tokens
    )

    summary = completion.choices[0].message.content.strip()

    append_Learning(session_id, session_file, "summary", summary)

    history.append({"role": "summary", "content": summary})
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return summary

def extract_keywords(text):
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 3]



"""
ATTENTION:
From down here is the Running proccess
"""



#Running Operation
def learning_core():
    client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY) #<-- Here If you wanna edit or change model!

    init_db()
    upgrade_memory_db()
    upgrade_learning_db()
    import_questions_json_to_db()
    init_db_learning()
    migrate_questions_json_to_db()
    while True:
        #Collecting Data on Memory
        session_id, session_file = ambil_sesi_belum_dipelajari()
        if not session_file:
            print("No sesion to Learning")
            break
        
        history = load_history(session_file)
        # Auto Extract from Keywords
        keywords = extract_keywords(" ".join([m["content"] for m in history]))
        related_history = []
        for kw in keywords:
            results = search_memory(kw, limit=3)
            for sid, role, content in results:
                related_history.append({"role": role, "content": content})
        #Token management
        if len(related_history) > 20:
            related_history = related_history[-20:]
        #Making Learning prompt
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        prompt = related_history + [
            {"role": "system", "content": "You are an extractor AI. Extract knowledge and make concise conclusions, keep only facts."},
            {"role": "user", "content": history_text}
        ]
        #Making conclusion from model
        completion = client.chat.completions.create(model=MODEL_NAME, messages=prompt)
        answer = completion.choices[0].message.content
        #Save to Learning DB
        append_Learning(session_id, session_file, "Thought", answer)
        #Brifing if too long answer
        summary = summarize_session(client, SUM_MODEL_NAME, session_id, session_file)
        #Generate Question
        if summary:
            q = generate_question(client, SUM_MODEL_NAME, summary)
            print("[QUESTION]", q)
            add_question_db(q)    #

        #Reset and Learning New topic
        prompt = []
        history = []
        #This function is work for canceling double thought by checking marks to topic
        tandai_sudah_dipelajari(session_id)
        print(f"Sesi {session_id} Learned")
        
        #Resting for cooling down NPU/GPU/CPU
        time.sleep(5)


#Running function, need checkup first 
def proccess_Running_AI_Online_mode(): 
    client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY)

    try:
        print("[INFO] Trying to connect to research mode...\n")
        rows = fetch_unanswered_questions(limit=10)
        if not rows:
            print("[INFO] No unanswered questions, switching to offline mode.")
            learning_core()
            return

        for qid, qtext in rows:
            print(f"[INFO] Researching: {qtext}")
            session_id, file_name = create_session("research_" + str(int(time.time())))
            intel = extract_intel_from_research(client, MODEL_NAME, session_id, file_name, qtext)

          
            conn = sqlite3.connect(DB_FILEL)
            c = conn.cursor()
            c.execute("UPDATE questions SET answered=1, answer=?, timestamp=? WHERE id=?", 
                      (intel, datetime.now().isoformat(), qid))
            conn.commit()
            conn.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        print("[INFO] Switching to Offline mode learning...\n")
        learning_core()


def Running_Learning_mode(): #<---incase you wanna run in terminal, run this definition to Running Learn mode
    #Function & Read Online check
    if check_internet():
        print("[INFO] Online detected, Starting Online mode Learning!")
        proccess_Running_AI_Online_mode()       
    else:
        print("[INFO] Offline mode, Starting Offline mode learning")
        learning_core()