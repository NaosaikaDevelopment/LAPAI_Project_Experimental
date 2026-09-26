from .state import *
def recall_from_faiss(query_vector, topk=5, threshold=cache.conf.get('thresholdrfs')):
    cache.faiss_index, cache.id_map

    if cache.faiss_index is None or cache.faiss_index.ntotal == 0:
        return []

    query_vector = np.asarray(query_vector,dtype="float32").reshape(1, -1)

    if query_vector.shape[1] != cache.faiss_index.d:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"query={query_vector.shape[1]}, index={cache.faiss_index.d}"
        )

    query_vector = normalize_embedding(query_vector)

    candidate_k = min(
        max(topk * 5, 30),
        cache.faiss_index.ntotal
    )

    scores, indices = cache.faiss_index.search(
        query_vector,
        candidate_k
    )

    results = []

    for similarity, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        similarity = float(similarity)

        if similarity < threshold:
            continue

        key = str(idx)

        if key not in cache.id_map:
            continue

        meta = cache.id_map[key]

        results.append({
            "uid": key,
            "content": meta.get("content", ""),
            "similarity": similarity
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results

def generate_embedding(text: str, embedding_type="passage"):
    if not isinstance(text, str):
        text = str(text)

    # E5 models are trained with query:/passage: prefixes.
    if embedding_type not in {"query", "passage"}:
        raise ValueError(
            "embedding_type must be 'query' or 'passage'."
        )

    model_input = f"{embedding_type}: {text}"

    tokens = cache.tokenizer(
        model_input,
        return_tensors="np",
        padding=True,
        truncation=True
    )
    if "token_type_ids" not in tokens:
        tokens["token_type_ids"] = np.zeros_like(
        tokens["input_ids"],
        dtype=np.int64
    )
    inputs = {k: v.astype("int64")for k, v in tokens.items()}

    outputs = cache.session.run(None, inputs)
    embedding = outputs[0]

    if embedding.ndim == 3:
        token_embeddings = embedding
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            mask = attention_mask[..., None].astype("float32")
            summed = (token_embeddings * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            embedding = summed / counts
        else:
            embedding = token_embeddings.mean(axis=1)

    embedding = np.asarray(
        embedding[0],
        dtype="float32"
    )

    return normalize_embedding(embedding)

def add_to_faiss(index,id_map,text,uid,meta_type="conversation",importance=1.0):
    vector = np.asarray(
        generate_embedding(text, embedding_type="passage"),
        dtype="float32"
    ).reshape(1, -1)

    if vector.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"vector={vector.shape[1]}, index={index.d}"
        )

    vector = normalize_embedding(vector)
    new_id = index.ntotal

    index.add(vector)

    id_map[str(new_id)] = {
        "uid": uid,
        "content": text,
        "meta_type": meta_type,
        "importance": float(importance),
        "timestamp": time.time()
    }

    faiss.write_index(index, cache.FAISS_INDEXM)

    with open(
        cache.FAISS_MAP,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            id_map,
            f,
            indent=2,
            ensure_ascii=False
        )

    return index, id_map

def normalize_embedding(vector):
    vector = np.asarray(
        vector,
        dtype="float32"
    )

    norm = np.linalg.norm(vector)

    if norm <= 0:
        raise ValueError("Embedding vector has zero norm.")

    return vector / norm

def init_faiss():

    cache.EMBED_DIM

    first_vec = generate_embedding("init", embedding_type="passage")
    embedding_dim = int(first_vec.shape[-1])

    if cache.EMBED_DIM != embedding_dim:
        print(
            f"[WARNING] Configured cache.EMBED_DIM={cache.EMBED_DIM}, "
            f"but embedding model produced {embedding_dim}. "
            f"Using {embedding_dim}."
        )
        cache.EMBED_DIM = embedding_dim

    if os.path.exists(cache.FAISS_INDEXM):
        print("[INFO] Loading existing FAISS index...")
        index = faiss.read_index(cache.FAISS_INDEXM)

        if not isinstance(index, faiss.IndexFlatIP):
            raise RuntimeError(
                "Existing FAISS index is not IndexFlatIP. "
                "Delete/rebuild Main_Memory.bin for normalized IP retrieval."
            )

        if index.d != cache.EMBED_DIM:
            raise RuntimeError(
                f"FAISS dimension mismatch: index={index.d}, "
                f"embedding={cache.EMBED_DIM}. "
                "Delete/rebuild Main_Memory.bin with the current embedding model."
            )

        if os.path.exists(cache.FAISS_MAP):
            with open(cache.FAISS_MAP, "r", encoding="utf-8") as f:
                id_map = json.load(f)
        else:
            id_map = {}
    else:
        print("[INFO] Creating new FAISS index...")
        index = faiss.IndexFlatIP(cache.EMBED_DIM)
        id_map = {}

        faiss.write_index(index, cache.FAISS_INDEXM)
        with open(cache.FAISS_MAP, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=2, ensure_ascii=False)

    if index.ntotal != len(id_map):
        print(
            f"[WARNING] FAISS/map mismatch: "
            f"index={index.ntotal}, map={len(id_map)}"
        )

    faiss_index = index
    return index, id_map





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
    
def add_personal_memory(text):
    if not text:
        return

    text = text.strip()

    # Prevent exact duplicate memories.
    for meta in cache.id_map.values():
        if (
            meta.get("meta_type") == "personal"
            and meta.get("content", "").strip().lower() == text.lower()
        ):
            return

    add_to_faiss(
        cache.faiss_index,
        cache.id_map,
        text,
        f"personal:{hash(text)}",
        meta_type="personal",
        importance=2.0
    )