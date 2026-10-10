from .state import *
from .memory import add_to_faiss, generate_embedding, recall_from_faiss


VALID_KINDS = {
    "question",
    "fact",
    "preference",
    "identity",
    "goal",
    "project",
    "event",
    "instruction",
    "small_talk",
}

VALID_MEMORY_KINDS = tuple(sorted(VALID_KINDS))

LONG_TERM_KINDS = {
    "fact",
    "preference",
    "identity",
    "goal",
    "project",
    "event",
}

MEMORY_CONFIDENCE_THRESHOLD = 0.70
RELATION_DECAY = 0.995



def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w]+", str(text).casefold(), flags=re.UNICODE)
        if len(token) >= 2
    }


def _safe_limit(value: Any, default: int = 5) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _recency_score(timestamp_text: str) -> float:
    timestamp = _parse_timestamp(timestamp_text)
    if timestamp <= 0:
        return 0.0
    age_days = max(0.0, (time.time() - timestamp) / 86400.0)
    return 1.0 / (1.0 + age_days / 30.0)


def _lexical_score(query_text: str, content: str) -> float:
    query_tokens = _token_set(query_text)
    content_tokens = _token_set(content)
    if not query_tokens:
        return 0.0
    coverage = len(query_tokens & content_tokens) / len(query_tokens)
    normalized_query = _normalize_text(query_text)
    normalized_content = _normalize_text(content)
    phrase_bonus = 0.15 if normalized_query and normalized_query in normalized_content else 0.0
    return min(1.0, coverage * 0.85 + phrase_bonus)


def _repair_long_term_fts(conn: sqlite3.Connection) -> None:
    """Repair missing/orphaned FTS rows without making FTS the source of truth."""
    try:
        missing = conn.execute(
            """
            SELECT m.id, m.text, m.category, m.kind, m.last_seen
            FROM long_term_memories AS m
            LEFT JOIN long_term_memories_fts AS f ON f.rowid = m.id
            WHERE f.rowid IS NULL
            """
        ).fetchall()
        for memory_id, text, category, kind, last_seen in missing:
            conn.execute(
                """
                INSERT INTO long_term_memories_fts
                    (rowid, memory_id, text, category, kind, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, str(memory_id), text, category, kind, last_seen),
            )

        conn.execute(
            """
            DELETE FROM long_term_memories_fts
            WHERE rowid NOT IN (SELECT id FROM long_term_memories)
            """
        )
    except sqlite3.OperationalError as exc:
        print(f"[WARNING] Long-term FTS repair skipped: {exc}")


def _safe_index_long_term_memory(memory_id: int, text: str, kind: str, category: str, confidence: float, session_id: Any) -> str | None:
    """Best-effort semantic index. SQLite remains authoritative."""
    try:
        if getattr(cache, "faiss_index", None) is None:
            return "FAISS index unavailable"
        add_to_faiss(
            cache.faiss_index,
            cache.id_map,
            text,
            f"long_term:{memory_id}",
            meta_type="long_term",
            importance=max(1.0, float(confidence)),
            session_id=session_id,
            memory_id=memory_id,
        )
        return None
    except Exception as exc:
        return str(exc)


def _row_to_memory(row) -> dict[str, Any]:
    (memory_id, session_id, text, kind, category, confidence, support_count, first_seen, last_seen,) = row
    return {
        "memory_id": int(memory_id),
        "session_id": session_id,
        "content": text,
        "kind": kind,
        "category": category,
        "confidence": float(confidence),
        "support_count": int(support_count),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "timestamp": _parse_timestamp(last_seen),
    }


def _utc_iso(value: str | None = None) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()

    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _safe_fts_query(text: str) -> str:
    # FTS5 gets a simple OR query of quoted tokens. This prevents raw
    # punctuation/operator input from becoming FTS syntax.
    terms = re.findall(r"[\w]+", str(text).casefold(), flags=re.UNICODE)
    terms = [term for term in terms if len(term) >= 2]

    if not terms:
        return ""

    return " OR ".join(
        '"' + term.replace('"', '""') + '"'
        for term in terms
    )


def init_memory_store() -> None:
    """Create and repair long-term-memory storage; never fail startup for stale FTS data."""
    if getattr(cache, "_memory_store_ready", False):
        return
    conn = sqlite3.connect(cache.DB_FILE, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS input_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                user_text TEXT NOT NULL,
                kind TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_input_classifications_session
            ON input_classifications(session_id, created_at)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                kind TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                support_count INTEGER NOT NULL DEFAULT 1,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
        """)
        try:
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_long_term_memory_exact
                ON long_term_memories(kind, category, normalized_text)
            """)
        except sqlite3.IntegrityError:
            print("[WARNING] Duplicate long-term memories detected; using non-unique index.")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_long_term_memory_exact
                ON long_term_memories(kind, category, normalized_text)
            """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_long_term_memories_category
            ON long_term_memories(category, last_seen)
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS long_term_memories_fts USING fts5(
                memory_id UNINDEXED,
                text,
                category,
                kind,
                last_seen UNINDEXED
            )
        """)
        _repair_long_term_fts(conn)
        conn.commit()
        cache._memory_store_ready = True
    except Exception as exc:
        conn.rollback()
        cache._memory_store_ready = False
        print(f"[WARNING] Memory store initialization failed: {exc}")
    finally:
        conn.close()



def _log_classification(conn: sqlite3.Connection,session_id: Any,user_text: str,kind: str,category: str,confidence: float,created_at: str,) -> None:
    conn.execute(
        """
        INSERT INTO input_classifications (
            session_id, user_text, kind, category, confidence, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_text,
            kind,
            category,
            confidence,
            created_at,
        ),
    )


def commit_memory(session_id: Any,user_text: str,kind: str,category: str,confidence: float,created_at: str | None = None,) -> dict[str, Any]:
    init_memory_store()

    if not isinstance(user_text, str) or not user_text.strip():
        return {"success": False, "stored": False, "error": "No current user input available."}

    kind = str(kind).strip().casefold()
    category = str(category).strip().casefold() or "general"
    timestamp = _utc_iso(created_at)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if kind not in VALID_KINDS:
        return {"success": False, "stored": False, "error": f"Invalid memory kind: {kind}"}

    normalized_text = _normalize_text(user_text)
    if not normalized_text:
        return {"success": False, "stored": False, "error": "Memory text is empty after normalization."}

    conn = sqlite3.connect(cache.DB_FILE, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        _log_classification(conn, session_id, user_text, kind, category, confidence, timestamp)

        if kind not in LONG_TERM_KINDS:
            conn.commit()
            return {
                "success": True,
                "stored": False,
                "kind": kind,
                "category": category,
                "confidence": confidence,
                "reason": "Input classified as non-long-term memory.",
            }

        if confidence < MEMORY_CONFIDENCE_THRESHOLD:
            conn.commit()
            return {
                "success": True,
                "stored": False,
                "kind": kind,
                "category": category,
                "confidence": confidence,
                "reason": "Confidence below memory threshold.",
            }

        # IMPORTANT: match the full uniqueness key. The old code matched only normalized_text and could incorrectly reinforce a memory with another kind/category.
        existing = conn.execute(
            """
            SELECT id, support_count, confidence, first_seen, last_seen
            FROM long_term_memories
            WHERE kind = ? AND category = ? AND normalized_text = ?
            ORDER BY support_count DESC, id ASC
            LIMIT 1
            """,
            (kind, category, normalized_text),
        ).fetchone()

        if existing:
            memory_id, support_count, old_confidence, first_seen, old_last_seen = existing
            new_support_count = int(support_count) + 1
            new_confidence = max(float(old_confidence), confidence)
            conn.execute(
                """
                UPDATE long_term_memories
                SET support_count = ?, confidence = ?, last_seen = ?
                WHERE id = ?
                """,
                (new_support_count, new_confidence, timestamp, memory_id),
            )
            conn.commit()
            return {
                "success": True,
                "stored": True,
                "updated": True,
                "memory_id": int(memory_id),
                "kind": kind,
                "category": category,
                "confidence": new_confidence,
                "support_count": new_support_count,
                "first_seen": first_seen,
                "last_seen": timestamp,
                "reason": "Existing exact memory reinforced.",
            }

        cursor = conn.execute(
            """
            INSERT INTO long_term_memories (
                session_id, text, normalized_text, kind, category,
                confidence, support_count, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (session_id, user_text, normalized_text, kind, category,
             confidence, timestamp, timestamp),
        )
        memory_id = int(cursor.lastrowid)

        conn.execute(
            """
            INSERT OR REPLACE INTO long_term_memories_fts
                (rowid, memory_id, text, category, kind, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (memory_id, str(memory_id), user_text, category, kind, timestamp),
        )
        conn.commit()

    except sqlite3.IntegrityError:
        conn.rollback()
        # Another writer may have inserted the exact same key. Resolve by re-reading the complete uniqueness key rather than guessing.
        existing = conn.execute(
            """
            SELECT id, support_count, confidence, first_seen, last_seen
            FROM long_term_memories
            WHERE kind = ? AND category = ? AND normalized_text = ?
            ORDER BY support_count DESC, id ASC
            LIMIT 1
            """,
            (kind, category, normalized_text),
        ).fetchone()
        if not existing:
            return {"success": False, "stored": False, "error": "Memory insert conflict could not be resolved."}

        memory_id, support_count, old_confidence, first_seen, _ = existing
        new_support_count = int(support_count) + 1
        new_confidence = max(float(old_confidence), confidence)
        conn.execute(
            """
            UPDATE long_term_memories
            SET support_count = ?, confidence = ?, last_seen = ?
            WHERE id = ?
            """,
            (new_support_count, new_confidence, timestamp, memory_id),
        )
        conn.commit()
        return {
            "success": True,
            "stored": True,
            "updated": True,
            "memory_id": int(memory_id),
            "kind": kind,
            "category": category,
            "confidence": new_confidence,
            "support_count": new_support_count,
            "first_seen": first_seen,
            "last_seen": timestamp,
            "reason": "Existing exact memory reinforced after insert race.",
        }
    except Exception as exc:
        conn.rollback()
        return {"success": False, "stored": False, "error": f"Memory database failure: {exc}"}
    finally:
        conn.close()

    result = {
        "success": True,
        "stored": True,
        "updated": False,
        "memory_id": memory_id,
        "kind": kind,
        "category": category,
        "confidence": confidence,
        "support_count": 1,
        "first_seen": timestamp,
        "last_seen": timestamp,
        "reason": "New long-term memory stored.",
    }
    index_warning = _safe_index_long_term_memory(
        memory_id, user_text, kind, category, confidence, session_id
    )
    if index_warning:
        result["index_warning"] = index_warning
    return result

def _parse_timestamp(timestamp_text: str) -> float:
    try:
        dt = datetime.fromisoformat(timestamp_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return 0.0


def recall_long_term_memory(user_input: str,limit: int = 5,threshold: float | None = None,) -> list[dict[str, Any]]:
    """Hybrid lexical + semantic long-term retrieval with fail-soft fallbacks."""
    try:
        init_memory_store()
        if not isinstance(user_input, str) or not user_input.strip():
            return []

        limit = _safe_limit(limit)
        query = _safe_fts_query(user_input)
        query_tokens = _token_set(user_input)

        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            rows = []
            if query:
                candidate_limit = max(limit * 8, 40)
                try:
                    rows = conn.execute(
                        """
                        SELECT m.id, m.session_id, m.text, m.kind, m.category,
                               m.confidence, m.support_count, m.first_seen,
                               m.last_seen, bm25(long_term_memories_fts) AS bm25_score
                        FROM long_term_memories_fts
                        JOIN long_term_memories AS m
                          ON m.id = long_term_memories_fts.rowid
                        WHERE long_term_memories_fts MATCH ?
                        ORDER BY bm25_score ASC, datetime(m.last_seen) DESC
                        LIMIT ?
                        """,
                        (query, candidate_limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    print(f"[WARNING] Long-term FTS retrieval failed: {exc}")
                    rows = []

            candidate_map: dict[int, dict[str, Any]] = {}
            bm25_values = [float(row[-1]) for row in rows] if rows else []
            best_bm25 = min(bm25_values) if bm25_values else 0.0
            worst_bm25 = max(bm25_values) if bm25_values else 0.0
            span = worst_bm25 - best_bm25

            for row in rows:
                item = _row_to_memory(row[:9])
                bm25_score = float(row[9])
                bm25_norm = 1.0 if span <= 1e-12 else (worst_bm25 - bm25_score) / span
                lexical = _lexical_score(user_input, item["content"])
                if query_tokens:
                    lexical = max(lexical, min(1.0, len(query_tokens & _token_set(item["content"])) / len(query_tokens)))
                item["fts_score"] = float(max(0.0, min(1.0, 0.75 * lexical + 0.25 * bm25_norm)))
                item["semantic_score"] = 0.0
                candidate_map[item["memory_id"]] = item

            # Semantic candidates extend recall beyond literal token overlap.
            try:
                query_vector = generate_embedding(user_input, embedding_type="query")
                semantic_hits = recall_from_faiss(
                    query_vector,
                    topk=max(limit * 8, 40),
                    threshold=0.0,
                    meta_type="long_term",
                )
                semantic_ids = [
                    int(hit["memory_id"])
                    for hit in semantic_hits
                    if hit.get("memory_id") is not None
                ]
                if semantic_ids:
                    placeholders = ",".join("?" for _ in semantic_ids)
                    extra_rows = conn.execute(
                        f"""
                        SELECT id, session_id, text, kind, category,
                               confidence, support_count, first_seen, last_seen
                        FROM long_term_memories
                        WHERE id IN ({placeholders})
                        """,
                        semantic_ids,
                    ).fetchall()
                    for row in extra_rows:
                        if int(row[0]) not in candidate_map:
                            candidate_map[int(row[0])] = _row_to_memory(row)

                semantic_by_id = {
                    int(hit["memory_id"]): float(hit.get("similarity", 0.0))
                    for hit in semantic_hits
                    if hit.get("memory_id") is not None
                }
            except Exception as exc:
                semantic_by_id = {}
                print(f"[WARNING] Long-term semantic retrieval unavailable: {exc}")

            ranked = []
            semantic_available = bool(semantic_by_id)
            for memory_id, item in candidate_map.items():
                item["semantic_score"] = float(semantic_by_id.get(memory_id, 0.0))
                lexical = float(item.get("fts_score", _lexical_score(user_input, item["content"])))
                confidence_score = max(0.0, min(1.0, float(item["confidence"])))
                support_score = min(1.0, int(item["support_count"]) / 5.0)
                recency = _recency_score(item["last_seen"])

                if semantic_available:
                    score = (
                        item["semantic_score"] * 0.50
                        + lexical * 0.30
                        + confidence_score * 0.10
                        + support_score * 0.05
                        + recency * 0.05
                    )
                else:
                    score = (
                        lexical * 0.65
                        + confidence_score * 0.15
                        + support_score * 0.10
                        + recency * 0.10
                    )

                item["recency_score"] = recency
                item["score"] = float(max(0.0, min(1.0, score)))
                ranked.append(item)
        finally:
            conn.close()

        if threshold is not None:
            try:
                threshold_value = float(threshold)
                ranked = [item for item in ranked if item["score"] >= threshold_value]
            except (TypeError, ValueError):
                pass

        ranked.sort(
            key=lambda item: (
                item["score"],
                item["support_count"],
                item["timestamp"],
            ),
            reverse=True,
        )
        selected = ranked[:limit]

        for item in selected:
            print(
                "[MEMORY RETRIEVAL]"
                f" score={item['score']:.4f}"
                f" semantic={item.get('semantic_score', 0.0):.4f}"
                f" lexical={item.get('fts_score', 0.0):.4f}"
                f" confidence={item['confidence']:.4f}"
                f" support={item['support_count']}"
                f" category={item['category']}"
                f" text={item['content'][:120]}"
            )

        return [
            {
                "role": "system",
                "content": item["content"],
                "score": item["score"],
                "semantic_score": item.get("semantic_score", 0.0),
                "relation_score": item.get("recency_score", 0.0),
                "fts_score": item.get("fts_score", 0.0),
                "confidence": item["confidence"],
                "support_count": item["support_count"],
                "kind": item["kind"],
                "category": item["category"],
                "date": item["last_seen"],
                "memory_id": item["memory_id"],
            }
            for item in selected
        ]
    except Exception as exc:
        print(f"[WARNING] Long-term memory recall failed: {exc}")
        return []

