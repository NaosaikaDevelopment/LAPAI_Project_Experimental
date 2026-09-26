from .state import *


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
    """Create isolated long-term-memory storage inside the existing memory DB."""
    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    c.execute("""
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

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_input_classifications_session
        ON input_classifications(session_id, created_at)
    """)

    c.execute("""
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

    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_long_term_memory_exact
        ON long_term_memories(kind, category, normalized_text)
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_long_term_memories_category
        ON long_term_memories(category, last_seen)
    """)

    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS long_term_memories_fts USING fts5(
            memory_id UNINDEXED,
            text,
            category,
            kind,
            last_seen UNINDEXED
        )
    """)

    conn.commit()
    conn.close()


def _log_classification(
    conn: sqlite3.Connection,
    session_id: Any,
    user_text: str,
    kind: str,
    category: str,
    confidence: float,
    created_at: str,
) -> None:
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
        return {
            "success": False,
            "stored": False,
            "error": "No current user input available.",
        }

    kind = str(kind).strip().casefold()
    category = str(category).strip().casefold() or "general"
    timestamp = _utc_iso(created_at)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if kind not in VALID_KINDS:
        return {
            "success": False,
            "stored": False,
            "error": f"Invalid memory kind: {kind}",
        }

    normalized_text = _normalize_text(user_text)

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        _log_classification(
            conn,
            session_id,
            user_text,
            kind,
            category,
            confidence,
            timestamp,
        )

        # Questions, instructions and small talk are deliberately kept in
        # the classification log but never enter long-term memory retrieval.
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

        existing = conn.execute(
            """
            SELECT id, support_count, confidence, first_seen
            FROM long_term_memories
            WHERE kind = ?
              AND category = ?
              AND normalized_text = ?
            LIMIT 1
            """,
            (kind, category, normalized_text),
        ).fetchone()

        if existing:
            memory_id, support_count, old_confidence, first_seen = existing
            new_support_count = int(support_count) + 1
            new_confidence = max(float(old_confidence), confidence)

            conn.execute(
                """
                UPDATE long_term_memories
                SET support_count = ?,
                    confidence = ?,
                    last_seen = ?
                WHERE id = ?
                """,
                (
                    new_support_count,
                    new_confidence,
                    timestamp,
                    memory_id,
                ),
            )
            conn.commit()

            return {
                "success": True,
                "stored": True,
                "updated": True,
                "memory_id": memory_id,
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
                session_id,
                text,
                normalized_text,
                kind,
                category,
                confidence,
                support_count,
                first_seen,
                last_seen
            )
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                session_id,
                user_text,
                normalized_text,
                kind,
                category,
                confidence,
                timestamp,
                timestamp,
            ),
        )

        memory_id = int(cursor.lastrowid)

        conn.execute(
            """
            INSERT INTO long_term_memories_fts (
                rowid, memory_id, text, category, kind, last_seen
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                str(memory_id),
                user_text,
                category,
                kind,
                timestamp,
            ),
        )

        conn.commit()

        return {
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

    except sqlite3.IntegrityError:
        conn.rollback()
        existing = conn.execute(
            """
            SELECT id, support_count, confidence, first_seen, last_seen
            FROM long_term_memories
            WHERE kind = ?
              AND category = ?
              AND normalized_text = ?
            LIMIT 1
            """,
            (kind, category, normalized_text),
        ).fetchone()

        if not existing:
            return {
                "success": False,
                "stored": False,
                "error": "Memory insert conflict could not be resolved.",
            }

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
            "memory_id": memory_id,
            "kind": kind,
            "category": category,
            "confidence": new_confidence,
            "support_count": new_support_count,
            "first_seen": first_seen,
            "last_seen": timestamp,
            "reason": "Existing exact memory reinforced after insert race.",
        }
    finally:
        conn.close()


def _parse_timestamp(timestamp_text: str) -> float:
    try:
        dt = datetime.fromisoformat(timestamp_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return 0.0


def recall_long_term_memory(user_input: str,limit: int = 5,threshold: float | None = None,) -> list[dict[str, Any]]:
    """
    Retrieve only classified long-term memories.

    FTS is the primary candidate generator. Candidates are grouped by
    category, ordered by newest-first, then receive a 0.5% per-result
    relation decay inside each category.
    """
    init_memory_store()

    query = _safe_fts_query(user_input)
    if not query:
        return []

    limit = max(1, int(limit))
    candidate_limit = max(limit * 5, 25)

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        rows = conn.execute(
            """
            SELECT
                m.id,
                m.session_id,
                m.text,
                m.kind,
                m.category,
                m.confidence,
                m.support_count,
                m.first_seen,
                m.last_seen,
                bm25(long_term_memories_fts) AS bm25_score
            FROM long_term_memories_fts
            JOIN long_term_memories AS m
              ON m.id = long_term_memories_fts.rowid
            WHERE long_term_memories_fts MATCH ?
            ORDER BY bm25_score ASC, m.last_seen DESC
            LIMIT ?
            """,
            (query, candidate_limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    if not rows:
        return []

    bm25_values = [float(row[-1]) for row in rows]
    best_bm25 = min(bm25_values)
    worst_bm25 = max(bm25_values)
    bm25_span = worst_bm25 - best_bm25
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        (
            memory_id,
            session_id,
            text,
            kind,
            category,
            confidence,
            support_count,
            first_seen,
            last_seen,
            bm25_score,
        ) = row

        bm25_score = float(bm25_score)
        if bm25_span <= 1e-12:
            fts_score = 1.0
        else:
            fts_score = (worst_bm25 - bm25_score) / bm25_span

        item = {
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
            "fts_score": fts_score,
        }
        groups.setdefault(category, []).append(item)

    ranked: list[dict[str, Any]] = []

    for category, items in groups.items():
        items.sort(
            key=lambda item: (
                item["timestamp"],
                item["support_count"],
            ),
            reverse=True,
        )

        for rank, item in enumerate(items):
            relation_score = RELATION_DECAY ** rank
            support_score = min(1.0, item["support_count"] / 5.0)

            final_score = (
                item["fts_score"] * 0.55
                + relation_score * 0.20
                + item["confidence"] * 0.15
                + support_score * 0.10
            )

            item["relation_score"] = relation_score
            item["score"] = float(final_score)
            ranked.append(item)

    if threshold is not None:
        ranked = [
            item
            for item in ranked
            if item["score"] >= float(threshold)
        ]

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["relation_score"],
            item["timestamp"],
        ),
        reverse=True,
    )

    selected = ranked[:limit]

    for item in selected:
        print(
            "[MEMORY RETRIEVAL]"
            f" score={item['score']:.4f}"
            f" fts={item['fts_score']:.4f}"
            f" relation={item['relation_score']:.4f}"
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
            "relation_score": item["relation_score"],
            "fts_score": item["fts_score"],
            "confidence": item["confidence"],
            "support_count": item["support_count"],
            "kind": item["kind"],
            "category": item["category"],
            "date": item["last_seen"],
            "memory_id": item["memory_id"],
        }
        for item in selected
    ]
