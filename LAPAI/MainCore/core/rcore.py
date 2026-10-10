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

def init_memory_state():
    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_state (
            session_id INTEGER PRIMARY KEY,
            summary TEXT DEFAULT '',
            summarized_until TEXT
        )
    """)

    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect(cache.DB_FILE)
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

    c.execute("""
    CREATE TABLE IF NOT EXISTS memory_state (
        session_id INTEGER PRIMARY KEY,
        summary TEXT DEFAULT '',
        summarized_until TEXT,
        summarized_until_rowid INTEGER DEFAULT 0
    )
    """)

    columns = {
        row[1]
        for row in c.execute(
            "PRAGMA table_info(memory_state)"
        ).fetchall()
    }

    if "summarized_until_rowid" not in columns:
        c.execute("""
            ALTER TABLE memory_state
            ADD COLUMN summarized_until_rowid INTEGER DEFAULT 0
        """)

    conn.commit()
    conn.close()

def init_learning_db():
    os.makedirs(cache.Learning_Dir, exist_ok=True)
    conn = sqlite3.connect(cache.Learning_DB)
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
        turn_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """)

    conn.commit()
    conn.close()
    

