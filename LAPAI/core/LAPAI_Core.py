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
from datetime import datetime
from transformers import AutoTokenizer






LEMONADE_BASE_URL = "http://localhost:8000/api/v1"
LEMONADE_API_KEY = "lemonade"

MODEL_NAME = "Llama-3.1-8B-Instruct-Hybrid" #<-- model you can change (in this case im using Lemonade server)
Sum_model = "Llama-3.2-3B-Instruct-Hybrid"

CHAT_DIR = "chats"
DB_FILE = "Raw_Memory.db"
Learning_Dir = "Learning"
FAISS_INDEX = "Main_Memory.bin"
FAISS_MAP = "memory_map.json"
QUESTIONS_FILE = "questions.json"
os.makedirs(CHAT_DIR, exist_ok=True)
model_path = "../all-mpnet-base-v2/onnx/model.onnx"
Learning_DB = os.path.join("Learning", "Learning.db")
tokenizer = AutoTokenizer.from_pretrained("../all-mpnet-base-v2")
session = ort.InferenceSession(model_path, providers=["DmlExecutionProvider"])

EMBED_DIM = 768 #<--- ATTENTION Embbeding dimension if you wanna change embedding model
faiss_index = None
id_map = {}










#---------------------------------MAIN FUNCTION-------------------------------
def init_faiss():
    global faiss_index, id_map
    if os.path.exists(FAISS_INDEX):
        print("[INFO] Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX)
        if os.path.exists(FAISS_MAP):
            with open(FAISS_MAP, "r", encoding="utf-8") as f:
                id_map = json.load(f)
        else:
            id_map = {}
    else:
        print("[INFO] Creating new FAISS index...")
        first_vec = generate_embedding("init")
        EMBEDDING_DIM = len(first_vec)
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        id_map={}
        faiss.write_index(index, FAISS_INDEX)
        with open(FAISS_MAP, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=2, ensure_ascii=False)
    faiss_index = index
    return index, id_map

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
        content,
        created_at UNINDEXED
    )
    """)

    conn.commit()
    conn.close()

def extract_keywords(text):
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 3]

def parse_yesno(user_it: str):
    match = re.search(r"\b(yes|no)\b", user_it, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "No"
    
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
    global faiss_index, id_map
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

    #Extended FAISS Function
    vector = generate_embedding(content)
    add_to_faiss(faiss_index, id_map, content, f"{session_id}:{role}:{len(content)}")


def search_memory(query, limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    SELECT session_id, role, content, bm25(messages) as score
    FROM messages
    WHERE messages MATCH ?
    ORDER BY score
    LIMIT ?
    """, (query, limit))
    results = c.fetchall()
    conn.close()
    return results

def recall_relevant_memory(user_input, limit=5, threshold=0.6):
    keywords = extract_keywords(user_input)
    if not keywords:
        return []
    LOTM = []
    query = " OR ".join(keywords)
    results = search_memory(query, limit=limit)
    for sid, role, content, score in results:
        if score <= threshold:
            LOTM.append({"role": role, "content": content})
    
    #FAIS Extended Function
    vector = generate_embedding(user_input)
    faiss_results = recall_from_faiss(vector, topk=limit)
    semantic_mem = [{"role": "system", "content": r} for r in faiss_results]

    return LOTM + semantic_mem

def load_history(json_file):
    if not os.path.exists(json_file):
        return []
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

#FAISS Function
def recall_from_faiss(query_vector, topk=5, threshold=0.4):
    global faiss_index, id_map

    if faiss_index is None or faiss_index.ntotal == 0:
        return []

    query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, topk)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        uid = str(idx)
        if uid in id_map:
            # convert L2 distance → similarity
            similarity = 1 / (1 + dist)  
            if similarity >= threshold:
                results.append({
                    "content": id_map[uid]["content"],
                    "score": float(similarity)
                })
    # urutkan berdasarkan skor
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def generate_embedding(text: str):
    tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    inputs = {k: v.astype("int64") for k, v in tokens.items()}
    outputs = session.run(None, inputs)


    embedding = outputs[0]  

    if embedding.ndim == 3:

        embedding = embedding.mean(axis=1)


    embedding = embedding[0]

    return embedding.astype("float32")

def add_to_faiss(index, id_map, text, uid):
    vector = generate_embedding(text)
    vector = np.array(vector, dtype="float32").reshape(1, -1)

    #print("DEBUG >> Vector shape:", vector.shape, "Index dim:", index.d) #Explanation \/
    """
    use this to check your embedding model vector shaped if you got error AssertionError. 
    for example you got debug DEBUG >> Vector shape: (1, 2304) Index dim: 768) 
    its mean your embedding model dimension is not sync, go change EMBEDDING_DIM on line 40, with:
    [Debug]
    Vector shape: (1, 2304) Index dim: 768) 
                      ↑ ↑ ↑
    This when you tested
    """
    new_id = len(id_map)
    index.add(vector)
    id_map[str(new_id)] = {"uid": uid, "content": text}

    # simpan ke file
    faiss.write_index(index, FAISS_INDEX)
    with open(FAISS_MAP, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)

    return index, id_map

    
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

#Learning feature in memory management
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
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

#Guard Scan
def estimate_tokens(messages):
    total = 0
    for m in messages:
        if isinstance(m, dict) and "content" in m:
            total += len(m["content"].split())
        elif isinstance(m, str):
            total += len(m.split())
    return total

#animtion
done = False  
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rStartup ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r                                                     \n') #ah dont mention it :V


#----------------------------------LEARNING FUNCTION-----------------------------------------

#Function

def init_learning_db():
    os.makedirs(Learning_Dir, exist_ok=True)
    conn = sqlite3.connect(Learning_DB)
    c = conn.cursor()


    c.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge USING fts5(
        topic,
        content,
        source,
        created_at
    )
    """)


    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        json_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)


    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT,
        content TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """)

    conn.commit()
    conn.close()



def create_session_Learning(title):
    conn = sqlite3.connect(Learning_DB)
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

#knowledge management
def search_knowledge(query, limit=5):
    conn = sqlite3.connect(Learning_DB)
    c = conn.cursor()
    sql = """
    SELECT topic, content, source, created_at, bm25(knowledge) as score
    FROM knowledge
    WHERE knowledge MATCH ?
    ORDER BY score
    LIMIT ?
    """
    c.execute(sql, (query, limit))
    rows = c.fetchall()
    conn.close()

    return [
        {"topic": r[0], "content": r[1], "source": r[2], "created_at": r[3], "score": r[4]}
        for r in rows
    ]

def load_knowledge(json_file):
    if not os.path.exists(json_file):
        return []
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def recall_knowledge(user_input, limit=5, threshold=0.6):
    keywords = extract_keywords(user_input)
    if not keywords:
        return []
    query = " OR ".join(keywords)

    results = search_knowledge(query, limit=limit)
    knowledge = []
    for r in results:
        if r["score"] <= threshold:
            knowledge.append({
                "role": "system",
                "content": f"[{r['topic']}] {r['content']} (src:{r['source']})",
                "score": r["score"]
            })
    return knowledge

def append_Learning(session_id, json_file, role, content):
    history = []
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append({"role": role, "content": content})
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    conn = sqlite3.connect(Learning_DB)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def start_learning(client, model, data_input, context):
    prompt = context
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role":"memory", "content":prompt},{"role":"Learning", "content": f"what you can learn from this user input: {data_input}"}]
    )
    return completion.choices[0].message.content




#--------------------------------Personal Function Information saver------------------------------
persnoal_file = "PersonalData.txt"
def init_personal_DB():
    if not os.path.exists(persnoal_file):
        with open(persnoal_file, "w", encoding="utf-8") as f:
            f.write("")
def append_txt(items):
    init_personal_DB()
    data_extract = f"[UserDataInformation: {items}]"
    with open(persnoal_file, "a", encoding="utf-8") as f:
        f.write(data_extract + "\n")
def read_txt():
    init_personal_DB()
    with open(persnoal_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]





#----------------------------------------------Oprational-----------------------------------------
"""
Explanation about this Oprational:

Main_Core_Function() its about self learing automation that learn from user and can provide an anser personal output
but the price for this function is kinda take a time, to decrease time to process from my suggestion by using hybrid model that provided by lemonade-server

Main_Core_FP_Function() its focused to FastResponse but keep the main thing like memory ability, knowledge recall, filtering, and kinda a lot thing that in there
but the price for this function its the system would not learning anything from user and any personal output, they all based from memory recall and personality injection,
but for API this function i recommended most, because its fast just one processing, and have some basic enhance ability.

if there you want adding your own function, just write down there, and i, i think i have already to tell each function just by the name.
Note from me:
Good luck and Have fun to experiment with it!
"""
def Main_Core_Function(user_msg, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile):
    client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY)
    done = False


    t = threading.Thread(target=animate)
    t.start()

    append_message(session_id, session_file, "user", user_msg)

    prompt =[]
    knowledge = []
    related_history = []
    LOTM = []
    new_items=[]

    keywords = extract_keywords(user_msg)
    collected_knowledge = recall_knowledge(user_msg)
    prompt.extend(knowledge)
    recalled = recall_relevant_memory(user_msg, limit=5)
    context = collected_knowledge + recalled + related_history
    context.sort(key=lambda x: x.get("score", 1.0), reverse=True)

    
    seen = set()
    for m in context:
        role = m.get("role", "")
        content = m.get("content", "")
        key = (role, str(content))  
        if key not in seen:
            item = {"role": role, "content": str(content)}
            if "score" in m:
                 item["score"] = m["score"]
            new_items.append(item)
            seen.add(key)
    prompt.extend(new_items)

    #Persona
    persona = ""
    if os.path.exists("PersonaAI.txt"):
        with open("PersonaAI.txt", "r", encoding="utf-8") as f:
            persona = f.read().strip()
    if persona:
        prompt.append({"role": "system", "content": persona})

    #Output
    try:  #<---Main model loading
        prompt.append({"role": "user", "content": user_msg})
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt
        )
        reply = completion.choices[0].message.content
        done = True
        t.join()
    except Exception as e:
        print("[ERROR] When Main Model loading:", str(e))
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system", "content": f"just Answer Yes or No. Is there any Personal Information in this input text: {user_msg}"}]
        )
        Decision = parse_yesno(completion.choices[0].message.content)
        if "Yes" in Decision:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"system", "content":f"just Answer short as possible, take the personal info from this input text: {user_msg}"}]
            )
            reject = ["No","no","No.","no.","NO","NO."]
            if any(q in completion.choices[0].message.content for q in reject):
                pass
            else:
                append_txt(completion.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR]: When try extract personal information: {e}")
    completion = client.chat.completions.create(
        model=Sum_model,
        messages=[{"role":"system", "content":f"Just answer what is actually meant by this user input: '{user_msg}'"}],
        max_tokens = 10
    )
    Thought = completion.choices[0].message.content
    append_Learning(seid, jsfile, "thought", Thought)
    append_message(session_id, session_file, "assistant", reply)
    summary = summarize_session(client, Sum_model, session_id, session_file)
    if summary:
        Learning_T = start_learning(client, MODEL_NAME, user_msg, prompt)
        question = generate_question(client, Sum_model, summary)
        add_question(question)
        append_Learning(seid, jsfile, "knowledge", Learning_T)
        append_message(session_id, session_file, "summary", summary)
    #Reset Token
    if estimate_tokens(prompt) >= 4056:
        print("Deleted Prompt")
        del prompt
    return reply
    
def Main_Core_FP_Function(user_msg, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile):
    client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY)
    done = False
    user_msg = msg
    t = threading.Thread(target=animate)
    t.start()
    append_message(session_id, session_file, "user", user_msg)
    prompt =[]
    knowledge = []
    related_history = []
    LOTM = []
    new_items=[]
    keywords = extract_keywords(user_msg)
    collected_knowledge = recall_knowledge(user_msg)
    prompt.extend(knowledge)
    recalled = recall_relevant_memory(user_msg, limit=5)
    context = collected_knowledge + recalled + related_history
    context.sort(key=lambda x: x.get("score", 1.0), reverse=True)

    seen = set()
    for m in context:
        role = m.get("role", "")
        content = m.get("content", "")
        key = (role, str(content))  
        if key not in seen:
            item = {"role": role, "content": str(content)}
            if "score" in m:
                 item["score"] = m["score"]
            new_items.append(item)
            seen.add(key)
    prompt.extend(new_items)

    #Persona
    persona = ""
    if os.path.exists("PersonaAI.txt"):
        with open("PersonaAI.txt", "r", encoding="utf-8") as f:
            persona = f.read().strip()
    if persona:
        prompt.append({"role": "system", "content": persona})

    #Output
    try:  #<---Main model loading
        prompt.append({"role": "user", "content": user_msg})
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt
        )
        reply = completion.choices[0].message.content
        done = True
        t.join()
    except Exception as e:
        print("[ERROR] When Main Model loading:", str(e))
    
    append_message(session_id, session_file, "assistant", reply)
    summary = summarize_session(client, Sum_model, session_id, session_file)
    if summary:
        append_message(session_id, session_file, "summary", summary)
    #Reset Token
    if estimate_tokens(prompt) >= 4056:
        print("Deleted Prompt")
        del prompt
    return reply
