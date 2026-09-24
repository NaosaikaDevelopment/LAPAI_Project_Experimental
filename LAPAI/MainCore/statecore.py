from .core import *
def _sqlite_timestamp_to_epoch(timestamp_text):
    if not timestamp_text:
        return time.time()

    try:
        dt = datetime.strptime(
            timestamp_text,
            "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)

        return dt.timestamp()
    except (TypeError, ValueError):
        return time.time()

def decsn(user_msg):
    completion = cache.client.chat.completions.create(
        model=cache.Sum_model,
        messages=[{"role":"system", "content": f"just Answer Yes or No. Is there any Personal Information in this input text: {user_msg}"}]
    )
    Decision = parse_yesno(completion.choices[0].message.content)
    if "Yes" in Decision:
        completion = cache.client.chat.completions.create(
            model=cache.Sum_model,
            messages=[{"role":"system", "content":f"just Answer short as possible, take the personal info from this input text: {user_msg}"}]
        )
        reject = ["No","no","No.","no.","NO","NO."]
        if any(q in completion.choices[0].message.content for q in reject):
            pass
        else:
            append_txt(completion.choices[0].message.content)
def append_txt(items):
    data_extract = f"[UserDataInformation: {items}]"

    with open(
        cache.persnoal_file,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(data_extract + "\n")

    try:
        add_personal_memory(data_extract)
    except Exception as e:
        print(f"[WARNING] Failed to index personal memory: {e}")
def read_txt():

    with open(cache.persnoal_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
def append_message(session_id, json_file, role, content):
    history = []
    cache.faiss_index, cache.id_map
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append({"role": role, "content": content})
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, datetime('now'))", (session_id, role, content))
    conn.commit()
    conn.close()

    #Extended FAISS Function

    importance = 1.0
    meta_type = "conversation"

    if role == "summary":
        add_to_faiss(
            cache.faiss_index,
            cache.id_map,
            content,
            f"{session_id}:{role}:{len(content)}",
            meta_type="summary",
            importance=compute_importance(content) * 1.5
        )

    add_to_faiss(
        cache.faiss_index,
        cache.id_map,
        content,
        f"{session_id}:{role}:{len(content)}",
        meta_type=meta_type,
        importance=compute_importance(content)
    )
def recall_relevant_memory(user_input, limit=10, threshold=cache.conf.get('thresholdrrm')):
    keywords = extract_keywords(user_input) or []

    fts_results = []

    if keywords:
        query = " OR ".join(
            f'"{keyword}"'
            for keyword in keywords
            if keyword
         )

    if query:
        fts_results = search_memory(
            query,
            limit=max(limit * 3, 20)
        )

    fts_results = search_memory(
        query,
        limit=limit
    )

    combined = []

    # FTS candidates
    for rowid, sid, role, content, created_at, bm25_score in fts_results:
        combined.append({
            "role": role,
            "content": content,
            "session_id": sid,
            "fts_score": normalize_fts(bm25_score),
            "faiss_score": 0.0,
            "importance": 1.0,
            "timestamp": _sqlite_timestamp_to_epoch(created_at),
            "rowid": rowid
        })

    # Semantic FAISS candidates.
    vector = generate_embedding(user_input,embedding_type="query")

    faiss_results = recall_from_faiss(vector,topk=max(limit * 3, 30),threshold=threshold)

    for result in faiss_results:
        key = result["uid"]
        meta = cache.id_map.get(key, {})

        type_weight = 1.0
        meta_type = meta.get("meta_type")

        if meta_type == "summary":
            type_weight = 1.2
        elif meta_type == "assistant_output":
            type_weight = 1.05

        combined.append({
            "role": "system",
            "content": result["content"],
            "session_id": None,
            "fts_score": 0.0,
            "faiss_score": float(result["similarity"]),
            "importance": float(
                meta.get("importance", 1.0)
            ) * type_weight,
            "timestamp": meta.get(
                "timestamp",
                time.time()
            )
        })

    scored = []

    for item in combined:
        recency = compute_recency(
            item["timestamp"]
        )

        role_weight = {
            "assistant": 1.1,
            "user": 1.0
        }.get(item["role"], 0.9)

        final_score = (
            item["fts_score"] * cache.conf.get('fscrItemFTS')
            + item["faiss_score"] * cache.conf.get('fscrItemFAISS')
            + item["importance"] * cache.conf.get('fscrItemImportance')
            + recency * cache.conf.get('fscrRecency')
            + role_weight * cache.conf.get('fscrRw')
        )

        item["score"] = float(final_score)
        scored.append(item)

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )
    for item in scored[:20]:
        #print(                                       #<---- Debuging ONLY
        #    f"score={item['score']:.3f}",
        #    f"fts={item['fts_score']:.3f}",
        #    f"faiss={item['faiss_score']:.3f}",
        #    item['content'][:120]
        #)
    return [
        {
            "role": item["role"],
            "content": item["content"],
            "score": item["score"]
        }
        for item in scored[:limit]
    ]
def initialize_core():
    """
    Initialize persistent storage, personal data, and FAISS.
    Call this once after importing LAPAI_Core.
    """
    init_db()
    init_memory_state()
    init_learning_db()
    title_hint = datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
    title_learn = "Learning"+title_hint
    cache.seid, cache.jsfile = create_session_Learning(title_learn)
    cache.faiss_index, cache.id_map = init_faiss()
    cache.session_id, cache.session_file = create_session(title_hint)
