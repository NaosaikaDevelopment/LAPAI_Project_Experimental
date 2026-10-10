from .state import *
def load_knowledge(json_file):
    if not os.path.exists(json_file):
        return []
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def search_knowledge(query, limit=5):
    conn = sqlite3.connect(cache.Learning_DB)
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
def extract_keywords(text):
    words = re.findall(r"\w+", text.lower())
    return words
def recall_knowledge(user_input, limit=5, threshold=0.6):
    keywords = extract_keywords(user_input)
    if not keywords:
        return []
    query = " OR ".join(keywords)

    results = search_knowledge(query, limit=limit)
    knowledge = []
    for r in results:
        if r["score"] <= threshold:
            trimmed_content = r["content"]
            if len(trimmed_content) > 800:
                trimmed_content = trimmed_content[:800]
            knowledge.append({
                "role": "system",
                "content": f"[{r['topic']}] {trimmed_content} (src:{r['source']})",
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

    conn = sqlite3.connect(cache.Learning_DB)
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