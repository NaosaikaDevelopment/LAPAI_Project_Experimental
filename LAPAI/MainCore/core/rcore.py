from .state import *        

def init_personal_DB():
    if not os.path.exists(cache.persnoal_file):
        with open(cache.persnoal_file, "w", encoding="utf-8") as f:
            f.write("")



def create_session(title):
    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", title) + ".json"
    json_file = os.path.join(cache.CHAT_DIR, filename)

    c.execute("INSERT INTO sessions (title, json_file) VALUES (?, ?)", (title, json_file))
    conn.commit()
    sid = c.lastrowid
    conn.close()

    if not os.path.exists(json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    return sid, json_file

def ctoolsSes(title):
    conn = sqlite3.connect(cache.TOOLSREADY)
    c = conn.cursor()

    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", title) + ".json"
    json_file = os.path.join(cache.TOOLS_DIR, filename)

    c.execute("INSERT INTO sessions (title, json_file) VALUES (?, ?)", (title, json_file))
    conn.commit()
    sid = c.lastrowid
    conn.close()

    if not os.path.exists(json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    return sid, json_file


def create_session_Learning(title):
    conn = sqlite3.connect(cache.Learning_DB)
    c = conn.cursor()

    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", title) + ".json"
    json_file = os.path.join(cache.Learning_Dir, filename)

    c.execute("INSERT INTO sessions (title, json_file) VALUES (?, ?)", (title, json_file))
    conn.commit()
    sid = c.lastrowid
    conn.close()

    if not os.path.exists(json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    return sid, json_file

os.makedirs(cache.CHAT_DIR, exist_ok=True)

