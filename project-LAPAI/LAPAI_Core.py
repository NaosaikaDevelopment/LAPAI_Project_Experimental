import os
import re
import json
import time
import sqlite3
import subprocess
from datetime import datetime
from openai import OpenAI






LEMONADE_BASE_URL = "http://localhost:8000/api/v1"
LEMONADE_API_KEY = "lemonade"
MODEL_NAME = "YourModel See at Terminal using 'Lemonade-server list' and copy it here" #<-- model you can change (in this case im using Lemonade server)
Sum_model = "Qwen2.5-0.5B-Instruct-CPU"
CHAT_DIR = "chats"
DB_FILE = "memory.db"
Learning_Dir = "Learning"
Learning_DB = os.path.join("Learning", "Learning.db")
os.makedirs(CHAT_DIR, exist_ok=True)
QUESTIONS_FILE = "questions.json"


#---------Function---------

def init_learning_db():
    os.makedirs("Learning", exist_ok=True)
    conn = sqlite3.connect(Learning_DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        content TEXT,
        source TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

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

def extract_keywords(text):
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 3]
    
#knowledge management
def search_knowledge(query, limit=5):
    conn = sqlite3.connect(Learning_DB)
    c = conn.cursor()
    c.execute("""
    SELECT topic, content, source, created_at
    FROM knowledge
    WHERE content LIKE ? OR topic LIKE ?
    ORDER BY id DESC
    LIMIT ?
    """, (f"%{query}%", f"%{query}%", limit))
    rows = c.fetchall()
    conn.close()

    return [{"role": "knowledge", "content": f"[{r[0]}] {r[1]} (src:{r[2]})"} for r in rows]

def load_knowledge(json_file):
    if not os.path.exists(json_file):
        return []
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def recall_knowledge(user_input, limit=5):
    keywords = extract_keywords(user_input)
    if not keywords:
        return []
    
    query = " OR ".join(keywords)
    
    return search_knowledge(query, limit=limit)

#memory management
def create_session(title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", title) + ".json"
    json_file = os.path.join(CHAT_DIR, filename)

    c.execute("INSERT INTO sessions (title, json_file) VALUES (?, ?)", (title, json_file))
    conn.commit()
    sid = c.lastrowid
    conn.close()

    if not os.path.exists(json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    return sid, json_file

def append_message(session_id, json_file, role, content):
    history = []
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append({"role": role, "content": content})
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    conn = sqlite3.connect(DB_FILE)
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

def recall_relevant_memory(user_input, limit=5):
    keywords = extract_keywords(user_input)
    if not keywords:
        return []
    
    query = " OR ".join(keywords)
    
    results = search_memory(query, limit=limit)
    
    LOTM = []
    for sid, role, content in results:
        LOTM.append({"role": role, "content": content})
    return LOTM

    
#Efesiensi Sumerize
def summarize_session(client, model, session_id, session_file, max_tokens=500):
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
            {"role": "system", "content": "You are a summarizer AI. Summarize this chat history in a concise way, keep facts, remove filler."},
            {"role": "user", "content": joined_text}
        ],
        max_tokens=max_tokens
    )

    summary = completion.choices[0].message.content.strip()

    with open(session_file, "w", encoding="utf-8") as f:
        new_data = [{"role": "summary", "content": summary}]
    if history:
        new_data.append(history[-1])
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return summary

#Learning feature 
def generate_question(client, model, text):
    prompt = f"Create a brief question about something odd or needing investigation from the following text: {text}\n\n"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()
def add_question(question):
    questions = load_json(QUESTIONS_FILE, [])
    if any(q["question"] == question for q in questions):
       return
    questions.append({
        "question": question,
        "answered": False,
        "timestamp": datetime.now().isoformat(),
        "answer": []
    })
    save_json(QUESTIONS_FILE, questions)
def load_json(path, default=[]):
    if not os.path.exists(path):
        return [] if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



#oprasional
def Tm():
    client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY)
    init_db()
    init_learning_db()



    print("ATTENTION first chat is making new sesion, and this is automatic generate title, keep in mind to asking something light and not take long context!")
    try:
        user_msg = input("\033[93m\nUser: \033[0m").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nKeluar.")
        exit()
    if user_msg.lower() in {"exit", "quit", "q"}:
        exit()

    completion = client.chat.completions.create(
        model=Sum_model,
        messages=[{"role": "system", "content": "Answear it as short as possible"},{"role": "user", "content": user_msg[:30]} ]
    )
    title_hint = completion.choices[0].message.content.strip() + datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
    session_id, session_file = create_session(title_hint)
    print("[Info] Session Created", title_hint, "\n")
       
    append_message(session_id, session_file, "user", user_msg)
    prompt=[]
    keywords = extract_keywords(user_msg)

    collected_knowledge = recall_knowledge(user_msg)
    knowledge = []
    knowledge.extend(collected_knowledge)
    prompt.extend(knowledge)
    
    prompt = [
        {"role": "system", "content": "Answear as short as possible"},
        {"role": "user", "content": user_msg}
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt
        )
        reply = completion.choices[0].message.content
        print("\033[92m\nLAPAI: \033[0m", end="", flush=True)
        for ch in reply:
            print(ch, end="", flush=True)
            time.sleep(0.02)
        print()
    except Exception as e:
        print("LAPAI: [error]", str(e))

    append_message(session_id, session_file, "assistant", reply)
    del prompt

    while True:
        try:
            user_msg = input("\033[93m\nUser: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar.")
            break

        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit", "q"}:
            break

        #knowledge
        append_message(session_id, session_file, "user", user_msg)
        prompt =[]
        keywords = extract_keywords(user_msg)
        collected_knowledge = recall_knowledge(user_msg)
        knowledge = []
        knowledge.extend(collected_knowledge)
        prompt.extend(knowledge)


        #Short term memory
        related_history = []
        seen = set()
        for kw in keywords:
            results = search_memory(kw, limit=3)
            for sid, role, content in results:
                if (role, content) not in seen:
                    related_history.append({"role": role, "content": content})
                    seen.add((role, content))

        #Long Term Memory Recall
        recalled = recall_relevant_memory(user_msg)
        LOTM = []
        LOTM.extend(recalled)
        persona = ""
        if os.path.exists("PersonaAI.txt"):
            with open("PersonaAI.txt", "r", encoding="utf-8") as f:
                persona = f.read().strip()
        prompt.extend(LOTM)
        prompt.extend(related_history)
        prompt.extend(knowledge)
        if persona:
            prompt.append({"role": "system", "content": persona})
        prompt.append({"role": "user", "content": user_msg})
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=prompt
            )

            reply = completion.choices[0].message.content

            print("\033[92m\nLAPAI: \033[0m", end="", flush=True)
            for ch in reply:
                print(ch, end="", flush=True)
                time.sleep(0.02)
            print()
        except Exception as e:
            print("LAPAI: [error]", str(e))
            continue

        append_message(session_id, session_file, "assistant", reply)
        summary = summarize_session(client, Sum_model, session_id, session_file)
        if summary:
            question = generate_question(client, Sum_model, summary)
            add_question(question)
            append_message(session_id, session_file, "summary", summary)
        #Reset Token
        del prompt
        


    subprocess.run("lemonade-server stop", shell=True)
