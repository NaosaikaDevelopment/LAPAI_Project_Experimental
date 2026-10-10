from .state import *
def recall_from_faiss(query_vector,topk=5,threshold=None,meta_type=None,session_id=None,):
    """Fail-soft FAISS retrieval.

    FAISS is an acceleration layer, not the source of truth. Any index/map
    problem returns an empty result so callers can continue with SQLite/FTS.
    """
    index = getattr(cache, "faiss_index", None)
    id_map_local = getattr(cache, "id_map", {}) or {}

    if index is None or getattr(index, "ntotal", 0) == 0:
        return []

    try:
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query_vector.shape[1] != index.d:
            print(
                "[WARNING] FAISS query dimension mismatch: "
                f"query={query_vector.shape[1]} index={index.d}"
            )
            return []

        query_vector = normalize_embedding(query_vector)
        try:
            threshold_value = float(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            threshold_value = None

        if meta_type is not None or session_id is not None:
            candidate_k = index.ntotal
        else:
            candidate_k = min(max(int(topk) * 5, 30), index.ntotal)

        scores, indices = index.search(query_vector, candidate_k)
    except Exception as exc:
        print(f"[WARNING] FAISS retrieval failed: {exc}")
        return []

    results = []
    for similarity, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        key = str(int(idx))
        meta = id_map_local.get(key)
        if not isinstance(meta, dict):
            continue

        similarity = float(similarity)
        if threshold_value is not None and similarity < threshold_value:
            continue
        if meta_type is not None and meta.get("meta_type") != meta_type:
            continue
        if session_id is not None and meta.get("session_id") != session_id:
            continue

        results.append({
            "uid": key,
            "content": meta.get("content", ""),
            "similarity": similarity,
            "meta_type": meta.get("meta_type"),
            "session_id": meta.get("session_id"),
            "memory_id": meta.get("memory_id"),
            "timestamp": meta.get("timestamp"),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max(1, int(topk))]


def generate_embedding(text: str, embedding_type="passage"):
    if not isinstance(text, str):
        text = str(text)

    if embedding_type not in {"query", "passage"}:
        raise ValueError("embedding_type must be 'query' or 'passage'.")

    model_input = f"{embedding_type}: {text}"
    tokens = cache.tokenizer(
        model_input,
        return_tensors="np",
        padding=True,
        truncation=True,
    )

    valid_inputs = {item.name for item in cache.session.get_inputs()}
    if "token_type_ids" in valid_inputs and "token_type_ids" not in tokens:
        tokens["token_type_ids"] = np.zeros_like(
            tokens["input_ids"],
            dtype=np.int64,
        )

    inputs = {
        key: value.astype("int64")
        for key, value in tokens.items()
        if key in valid_inputs
    }
    if not inputs:
        raise RuntimeError("Embedding model accepted no tokenizer inputs.")

    outputs = cache.session.run(None, inputs)
    if not outputs:
        raise RuntimeError("Embedding model returned no outputs.")

    embedding = np.asarray(outputs[0])
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

    if embedding.ndim == 2:
        embedding = embedding[0]
    elif embedding.ndim != 1:
        raise RuntimeError(
            f"Unexpected embedding output shape: {tuple(embedding.shape)}"
        )

    embedding = np.asarray(embedding, dtype="float32")
    return normalize_embedding(embedding)


def add_to_faiss(index,id_map,text,uid,meta_type="conversation",importance=1.0,session_id=None,memory_id=None,):
    """Add one vector and persist it; persistence failures never change the DB source of truth."""
    vector = np.asarray(
        generate_embedding(text, embedding_type="passage"),
        dtype="float32",
    ).reshape(1, -1)

    if vector.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: vector={vector.shape[1]}, index={index.d}"
        )

    vector = normalize_embedding(vector)
    new_id = index.ntotal
    index.add(vector)

    id_map[str(new_id)] = {
        "uid": uid,
        "content": text,
        "meta_type": meta_type,
        "importance": float(importance),
        "timestamp": time.time(),
        "session_id": session_id,
        "memory_id": memory_id,
    }

    # Use unique temp files so concurrent calls do not overwrite one another.
    import tempfile
    index_tmp = None
    map_tmp = None
    try:
        index_dir = os.path.dirname(cache.FAISS_INDEXM) or "."
        map_dir = os.path.dirname(cache.FAISS_MAP) or "."
        fd, index_tmp = tempfile.mkstemp(prefix="faiss_", suffix=".tmp", dir=index_dir)
        os.close(fd)
        fd, map_tmp = tempfile.mkstemp(prefix="faissmap_", suffix=".tmp", dir=map_dir)
        os.close(fd)

        faiss.write_index(index, index_tmp)
        with open(map_tmp, "w", encoding="utf-8") as handle:
            json.dump(id_map, handle, indent=2, ensure_ascii=False)
        os.replace(index_tmp, cache.FAISS_INDEXM)
        index_tmp = None
        os.replace(map_tmp, cache.FAISS_MAP)
        map_tmp = None
    except Exception:
        # Keep the in-memory index/map consistent with the persisted state.
        try:
            index.remove_ids(np.asarray([new_id], dtype="int64"))
        except Exception:
            pass
        id_map.pop(str(new_id), None)
        raise
    finally:
        for tmp in (index_tmp, map_tmp):
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    return index, id_map


def normalize_embedding(vector):
    vector = np.asarray(vector, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("Embedding vector has zero norm.")
    return vector / norm


def init_faiss():
    """Load FAISS when healthy; otherwise start with an empty recoverable index."""
    first_vec = generate_embedding("init", embedding_type="passage")
    embedding_dim = int(first_vec.shape[-1])

    configured_dim = getattr(cache, "EMBED_DIM", None)
    if configured_dim != embedding_dim:
        print(
            f"[WARNING] Configured EMBED_DIM={configured_dim}, "
            f"model produced {embedding_dim}; using {embedding_dim}."
        )
        cache.EMBED_DIM = embedding_dim

    index = None
    if os.path.exists(cache.FAISS_INDEXM):
        try:
            candidate = faiss.read_index(cache.FAISS_INDEXM)
            if not isinstance(candidate, faiss.IndexFlatIP):
                raise RuntimeError("not IndexFlatIP")
            if candidate.d != embedding_dim:
                raise RuntimeError(
                    f"dimension mismatch index={candidate.d} expected={embedding_dim}"
                )
            index = candidate
            print("[INFO] Loading existing FAISS index...")
        except Exception as exc:
            print(f"[WARNING] Existing FAISS index is unusable: {exc}")
            try:
                broken = f"{cache.FAISS_INDEXM}.broken"
                if os.path.exists(broken):
                    os.remove(broken)
                os.replace(cache.FAISS_INDEXM, broken)
            except OSError:
                pass

    if index is None:
        print("[INFO] Creating new FAISS index...")
        index = faiss.IndexFlatIP(embedding_dim)

    id_map = {}
    if os.path.exists(cache.FAISS_MAP):
        try:
            with open(cache.FAISS_MAP, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                id_map = {
                    str(key): value
                    for key, value in loaded.items()
                    if str(key).isdigit() and int(key) < index.ntotal and isinstance(value, dict)
                }
        except Exception as exc:
            print(f"[WARNING] Failed to load FAISS map; starting clean map: {exc}")

    if index.ntotal != len(id_map):
        print(
            f"[WARNING] FAISS/map mismatch remains: index={index.ntotal}, map={len(id_map)}. "
            "Missing map rows will be ignored until rebuilt."
        )

    try:
        faiss.write_index(index, cache.FAISS_INDEXM)
        with open(cache.FAISS_MAP, "w", encoding="utf-8") as handle:
            json.dump(id_map, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[WARNING] Could not persist initial FAISS state: {exc}")

    return index, id_map

def add_personal_memory(text):
    if not text:
        return

    text = text.strip()

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