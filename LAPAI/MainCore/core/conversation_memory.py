from __future__ import annotations

from .state import *
from datetime import date
from .memory import add_to_faiss, generate_embedding, recall_from_faiss
from .promptmanager import get_prompt_manager
from .addonsfunction import compute_importance
from .memorystore import recall_long_term_memory, _safe_fts_query


DEFAULT_ACTIVE_TOPIC_THRESHOLD = cache.conf.get('ActTopicThr')
DEFAULT_SUMMARY_RECALL_THRESHOLD = cache.conf.get('SumReclThr')
DEFAULT_ACTIVE_MEMORY_TOKEN_LIMIT = cache.conf.get('ActMemTokLim')
DEFAULT_DETAIL_RECALL_LIMIT = cache.conf.get('DetReclLim')
DEFAULT_SUMMARY_RECALL_LIMIT = cache.conf.get('SumRecLim')


MONTHS = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    # Indonesian
    "januari": 1, "februari": 2, "maret": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
    "september": 9, "oktober": 10, "november": 11, "desember": 12,
    # German
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
def _conf(key: str, default: Any) -> Any:
    value = cache.conf.get(key, default)
    return default if value is None else value


def init_conversation_memory() -> None:
    conn = sqlite3.connect(cache.DB_FILE)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS active_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'trimmed', 'archived'))
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_active_memory_session_status
            ON active_memory(session_id, status, created_at)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                topic TEXT NOT NULL DEFAULT 'general',
                summary TEXT NOT NULL,
                source_count INTEGER NOT NULL DEFAULT 0,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_summaries_session
            ON conversation_summaries(session_id, last_seen)
            """
        )

        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS conversation_summaries_fts USING fts5(
                summary_id UNINDEXED,
                topic,
                summary,
                last_seen UNINDEXED
            )
            """
        )
        conn.commit()
    finally:
        conn.close()




def migrate_legacy_summaries() -> int:
    init_conversation_memory()
    conn = sqlite3.connect(cache.DB_FILE)
    try:
        rows = conn.execute(
            """
            SELECT session_id, summary, summarized_until
            FROM memory_state
            WHERE summary IS NOT NULL AND trim(summary) != ''
            """
        ).fetchall()

        migrated = 0
        for session_id, summary, summarized_until in rows:
            exists = conn.execute(
                """
                SELECT 1
                FROM conversation_summaries
                WHERE session_id = ? AND summary = ?
                LIMIT 1
                """,
                (session_id, summary),
            ).fetchone()
            if exists:
                continue

            timestamp = _parse_dt(summarized_until).isoformat() if summarized_until else datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                INSERT INTO conversation_summaries (
                    session_id, topic, summary, source_count,
                    first_seen, last_seen, created_at
                )
                VALUES (?, 'legacy', ?, 0, ?, ?, ?)
                """,
                (session_id, summary, timestamp, timestamp, datetime.now(timezone.utc).isoformat()),
            )
            summary_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO conversation_summaries_fts (
                    rowid, summary_id, topic, summary, last_seen
                )
                VALUES (?, ?, 'legacy', ?, ?)
                """,
                (summary_id, str(summary_id), summary, timestamp),
            )
            migrated += 1

    finally:
        conn.commit()
        conn.close()

    return migrated

def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return datetime.now(timezone.utc)


def _parse_date_hint(text: str) -> date | None:
    text = str(text or "")

    # ISO / slash / dash dates.
    for pattern in (
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b",
    ):
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            groups = [int(value) for value in match.groups()]
            if groups[0] > 1900:
                year, month, day = groups
            else:
                day, month, year = groups
            return datetime(year, month, day).date()
        except ValueError:
            continue

    month_pattern = r"(?:\b(\d{1,2})\s+([\wÄÖÜäöüß]+)\s+(20\d{2})\b|\b([\wÄÖÜäöüß]+)\s+(\d{1,2})(?:,)?\s+(20\d{2})\b)"
    match = re.search(month_pattern, text, flags=re.IGNORECASE)
    if match:
        d1, m1, y1, m2, d2, y2 = match.groups()
        try:
            if d1:
                day, month_name, year = int(d1), m1.casefold(), int(y1)
            else:
                day, month_name, year = int(d2), m2.casefold(), int(y2)
            month = MONTHS.get(month_name)
            if month:
                return datetime(year, month, day).date()
        except ValueError:
            pass

    return None


def _date_distance(item: dict[str, Any], target_date: datetime.date) -> float:
    first = _parse_dt(item.get("first_seen")).date()
    last = _parse_dt(item.get("last_seen")).date()

    if first <= target_date <= last:
        return 0.0
    return float(min(abs((target_date - first).days), abs((target_date - last).days)))


def _active_token_limit() -> int:
    configured = _conf("activeMemoryTokenLimit", DEFAULT_ACTIVE_MEMORY_TOKEN_LIMIT)
    return max(256, int(configured))


def _active_topic_threshold() -> float:
    value = float(_conf("activeTopicThreshold", DEFAULT_ACTIVE_TOPIC_THRESHOLD))
    return min(max(value, 0.0), 0.99)


def _summary_recall_threshold() -> float:
    value = float(_conf("summaryRecallThreshold", DEFAULT_SUMMARY_RECALL_THRESHOLD))
    return min(max(value, 0.0), 0.99)


def _recall_query_text(session_id: int,user_input: str,max_turns: int = 2,max_chars: int = 300,) -> str:
    try:
        rows = _active_rows(session_id)
    except Exception:
        rows = []

    recent_texts = [row[1] for row in rows[-max_turns:] if row[1]]
    context = " ".join(recent_texts)[-max_chars:]

    if not context:
        return user_input

    return f"{context} {user_input}".strip()

def _active_rows(session_id: int) -> list[tuple]:
    init_conversation_memory()
    conn = sqlite3.connect(cache.DB_FILE)
    try:
        return conn.execute(
            """
            SELECT id, user_content, assistant_content, created_at
            FROM active_memory
            WHERE session_id = ? AND status = 'active'
            ORDER BY datetime(created_at) ASC, id ASC
            """,
            (session_id,),
        ).fetchall()
    finally:
        conn.close()


def _active_row_by_id(row_id: int):
    conn = sqlite3.connect(cache.DB_FILE)
    try:
        return conn.execute(
            """
            SELECT id, session_id, user_content, assistant_content, created_at, status
            FROM active_memory
            WHERE id = ?
            """,
            (row_id,),
        ).fetchone()
    finally:
        conn.close()


def _turn_messages(rows: list[tuple]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for _, user_content, assistant_content, _ in rows:
        messages.append({"role": "user", "content": user_content})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
    return messages


def _mark_status(row_ids: list[int], status: str) -> None:
    if not row_ids:
        return
    conn = sqlite3.connect(cache.DB_FILE)
    try:
        placeholders = ",".join("?" for _ in row_ids)
        conn.execute(
            f"UPDATE active_memory SET status = ? WHERE id IN ({placeholders})",
            [status, *row_ids],
        )
        conn.commit()
    finally:
        conn.close()


def _summarize_archived_turns(
    session_id: int,
    rows: list[tuple],
) -> dict[str, Any] | None:
    if not rows:
        return None

    transcript = []
    for _, user_content, assistant_content, created_at in rows:
        transcript.append(
            f"[{created_at}] USER: {user_content}\n"
            f"[{created_at}] ASSISTANT: {assistant_content}"
        )

    prompt = f"""
You are writing a persistent conversation memory summary.

Conversation segment to archive:
{chr(10).join(transcript)}

Create one compact summary that preserves information useful in future turns:
- the actual topic
- important facts and decisions
- ongoing technical/project context
- unresolved issues
- user preferences or constraints when relevant
- concrete dates when they matter

Remove greetings, filler, repetition, and temporary chatter.

Return exactly two lines:
TOPIC: <short topic name>
SUMMARY: <compact persistent summary>
"""

    completion = cache.client.chat.completions.create(
        model=cache.Sum_model,
        messages=[
            {
                "role": "system",
                "content": "Maintain concise persistent conversation memory.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=int(_conf("comMaxSumToken", 500)),
    )

    raw = (completion.choices[0].message.content or "").strip()
    topic = "general"
    summary = raw

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("TOPIC:"):
            topic = line.split(":", 1)[1].strip() or "general"
        elif line.upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    if not summary:
        return None

    first_seen = min(
        (_parse_dt(row[3]) for row in rows),
        default=datetime.now(timezone.utc),
    ).isoformat()
    last_seen = max(
        (_parse_dt(row[3]) for row in rows),
        default=datetime.now(timezone.utc),
    ).isoformat()
    created_at = datetime.now(timezone.utc).isoformat()

    init_conversation_memory()
    conn = sqlite3.connect(cache.DB_FILE)
    try:
        cursor = conn.execute(
            """
            INSERT INTO conversation_summaries (
                session_id, topic, summary, source_count,
                first_seen, last_seen, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                topic,
                summary,
                len(rows),
                first_seen,
                last_seen,
                created_at,
            ),
        )
        summary_id = int(cursor.lastrowid)

        conn.execute(
            """
            INSERT INTO conversation_summaries_fts (
                rowid, summary_id, topic, summary, last_seen
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                summary_id,
                str(summary_id),
                topic,
                summary,
                last_seen,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    try:
        add_to_faiss(
            cache.faiss_index,
            cache.id_map,
            f"TOPIC: {topic}\nSUMMARY: {summary}",
            f"summary:{summary_id}",
            meta_type="sum",
            importance=compute_importance(summary) * 1.5,
            session_id=session_id,
            memory_id=summary_id,
        )
    except Exception as exc:
        print(f"[WARNING] Failed to index conversation summary: {exc}")

    return {
        "summary_id": summary_id,
        "topic": topic,
        "summary": summary,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


def _is_short_continuation(user_input: str) -> bool:
    terms = re.findall(r"[\w]+", str(user_input or ""), flags=re.UNICODE)
    if len(terms) > 7:
        return False

    continuation_terms = {
        "lanjut", "terus", "kemudian", "lalu", "itu", "ini", "bagaimana",
        "kenapa", "mengapa", "detail", "jelaskan", "iya", "ya", "yes",
        "okay", "oke", "selanjutnya", "weiter", "genau", "und", "also",
    }
    return any(term.casefold() in continuation_terms for term in terms)


def prepare_active_memory(session_id: int, user_input: str) -> list[dict[str, str]]:
    init_conversation_memory()
    rows = _active_rows(session_id)
    if not rows:
        return []

    query_vector = np.asarray(
        generate_embedding(user_input, embedding_type="query"),
        dtype="float32",
    ).reshape(1, -1)

    related_rows: list[tuple] = []
    archived_rows: list[tuple] = []
    topic_threshold = _active_topic_threshold()
    if _is_short_continuation(user_input):
        topic_threshold = min(topic_threshold, 0.25)

    for row in rows:
        row_id, user_content, assistant_content, _ = row
        turn_text = f"USER: {user_content}\nASSISTANT: {assistant_content}"
        vector = np.asarray(
            generate_embedding(turn_text, embedding_type="passage"),
            dtype="float32",
        ).reshape(1, -1)
        query_vec = _normalize(query_vector[0])
        target_vec = _normalize(vector[0])
        score = float(np.dot(query_vec[0], target_vec[0]))

        if score >= topic_threshold:
            related_rows.append(row)
        else:
            archived_rows.append(row)

        print(
            f"[ACTIVE MEMORY] id={row_id} topic_score={score:.4f} "
            f"threshold={topic_threshold:.4f} "
            f"status={'keep' if score >= topic_threshold else 'archive'}"
        )

    if archived_rows:
        try:
            _summarize_archived_turns(session_id, archived_rows)
        finally:
            _mark_status([row[0] for row in archived_rows], "archived")

    manager = get_prompt_manager()
    while related_rows and manager.estimate_tokens(_turn_messages(related_rows)) > _active_token_limit():
        oldest = related_rows.pop(0)
        _mark_status([oldest[0]], "trimmed")

    return _turn_messages(related_rows)


def record_active_turn(
    session_id: int,
    user_content: str,
    assistant_content: str,
    created_at: str | None = None,
) -> int:
    """Persist one completed user+assistant turn into active memory."""
    init_conversation_memory()
    timestamp = _parse_dt(created_at).isoformat() if created_at else datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        cursor = conn.execute(
            """
            INSERT INTO active_memory (
                session_id, user_content, assistant_content, created_at, status
            )
            VALUES (?, ?, ?, ?, 'active')
            """,
            (session_id, str(user_content), str(assistant_content), timestamp),
        )
        row_id = int(cursor.lastrowid)
        conn.commit()
    finally:
        conn.close()

    try:
        add_to_faiss(
            cache.faiss_index,
            cache.id_map,
            f"USER: {user_content}\nASSISTANT: {assistant_content}",
            f"active:{row_id}",
            meta_type="active",
            importance=compute_importance(user_content + " " + assistant_content),
            session_id=session_id,
            memory_id=row_id,
        )
    except Exception as exc:
        print(f"[WARNING] Failed to index active memory: {exc}")

    return row_id


def _summary_hits(session_id: int,user_input: str,limit: int,threshold: float | None = None,) -> list[dict[str, Any]]:
    init_conversation_memory()
    if cache.faiss_index is None:
        return []

    query_text = _recall_query_text(session_id, user_input)
    vector = generate_embedding(query_text, embedding_type="query")
    date_hint = _parse_date_hint(user_input)
    selected = []
    seen = set()

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        if date_hint:
            # A specific date is authoritative. Retrieve summaries whose
            # covered time range contains that date before semantic ranking.
            rows = conn.execute(
                """
                SELECT id, topic, summary, first_seen, last_seen, source_count
                FROM conversation_summaries
                WHERE date(first_seen) <= date(?)
                  AND date(last_seen) >= date(?)
                ORDER BY datetime(last_seen) DESC
                LIMIT ?
                """,
                (date_hint.isoformat(), date_hint.isoformat(), max(limit * 5, 15)),
            ).fetchall()

            for row in rows:
                summary_vector = np.asarray(
                    generate_embedding(
                        f"TOPIC: {row[1]}\nSUMMARY: {row[2]}",
                        embedding_type="passage",
                    ),
                    dtype="float32",
                )
                query_vec = _normalize(summary_vector[0])
                target_vec = _normalize(vector[0])
                similarity = float(np.dot(query_vec, target_vec))
                selected.append({
                    "summary_id": int(row[0]),
                    "topic": row[1],
                    "content": row[2],
                    "first_seen": row[3],
                    "last_seen": row[4],
                    "source_count": int(row[5]),
                    "similarity": similarity,
                })
                seen.add(int(row[0]))

        if not selected:
            results = recall_from_faiss(
                vector,
                topk=max(limit * 4, 12),
                threshold=_summary_recall_threshold() if threshold is None else threshold,
                meta_type="sum",
            )

            for item in results:
                memory_id = item.get("memory_id")
                if memory_id is None or memory_id in seen:
                    continue

                row = conn.execute(
                    """
                    SELECT id, topic, summary, first_seen, last_seen, source_count
                    FROM conversation_summaries
                    WHERE id = ?
                    """,
                    (int(memory_id),),
                ).fetchone()
                if row is None:
                    continue

                seen.add(int(memory_id))
                selected.append({
                    "summary_id": int(row[0]),
                    "topic": row[1],
                    "content": row[2],
                    "first_seen": row[3],
                    "last_seen": row[4],
                    "source_count": int(row[5]),
                    "similarity": float(item.get("similarity", 0.0)),
                })
    finally:
        conn.close()

    if date_hint:
        selected.sort(
            key=lambda item: (
                _date_distance(item, date_hint) == 0,
                -_date_distance(item, date_hint),
                item["similarity"],
            ),
            reverse=True,
        )
    else:
        selected.sort(
            key=lambda item: (
                item["similarity"],
                _parse_dt(item["last_seen"]).timestamp(),
            ),
            reverse=True,
        )

    return selected[:max(1, int(limit))]


def has_summary_memory(session_id: int, user_input: str) -> bool:
    """Return whether the current input matches an archived topic summary."""
    return bool(_summary_hits(session_id, user_input, limit=1))


def recall_detailed_memory(session_id: int,user_input: str,limit: int = DEFAULT_DETAIL_RECALL_LIMIT,) -> list[dict[str, Any]]:
    query = _safe_fts_query(user_input)
    if not query:
        return []

    target_date = _parse_date_hint(user_input)
    candidate_limit = max(int(limit) * 8, 40)

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        params: list[Any] = [query]
        date_clause = ""
        if target_date:
            date_clause = "AND date(created_at) = ?"
            params.append(target_date.isoformat())

        params.append(candidate_limit)

        rows = conn.execute(
            f"""
            SELECT rowid, session_id, role, content, created_at,
                   bm25(messages) AS bm25_score
            FROM messages
            WHERE messages MATCH ?
              AND role IN ('user', 'assistant')
              {date_clause}
            ORDER BY bm25_score ASC, datetime(created_at) DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        if not rows and target_date:
            rows = conn.execute(
                """
                SELECT rowid, session_id, role, content, created_at,
                       0.0 AS bm25_score
                FROM messages
                WHERE role IN ('user', 'assistant')
                  AND date(created_at) = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (target_date.isoformat(), candidate_limit),
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    if not rows:
        return []

    bm25 = [float(row[5]) for row in rows]
    best = min(bm25)
    worst = max(bm25)
    span = worst - best

    now_ts = datetime.now(timezone.utc).timestamp()
    results = []
    for row in rows:
        rowid, row_session_id, role, content, created_at, bm25_score = row
        if len(content) > 800:
            content = content[:800]
        fts_score = 1.0 if span <= 1e-12 else (worst - float(bm25_score)) / span
        timestamp = _parse_dt(created_at).timestamp()
        age_days = max(0.0, (now_ts - timestamp) / 86400.0)
        recency_score = 1.0 / (1.0 + age_days)

        if target_date:
            date_score = 1.0 if _parse_dt(created_at).date() == target_date else 0.0
            final_score = date_score * 0.65 + fts_score * 0.35
        else:
            final_score = fts_score * 0.60 + recency_score * 0.40

        results.append({
            "role": role,
            "content": content,
            "date": created_at,
            "session_id": row_session_id,
            "fts_score": float(fts_score),
            "recency_score": float(recency_score),
            "score": float(final_score),
            "rowid": int(rowid),
        })

    results.sort(
        key=lambda item: (
            item["score"],
            _parse_dt(item["date"]).timestamp(),
        ),
        reverse=True,
    )

    return results[:max(1, int(limit))]


def recall_memory_context(session_id: int,user_input: str,summary_limit: int = DEFAULT_SUMMARY_RECALL_LIMIT,detail_limit: int = DEFAULT_DETAIL_RECALL_LIMIT,include_long_term: bool = True,) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Two-stage memory recall:
      1. topic summary recall
      2. detailed ranking only when a relevant summary exists
    """
    summaries = _summary_hits(session_id, user_input, summary_limit)
    detailed: list[dict[str, Any]] = []

    if summaries:
        detailed.extend(recall_detailed_memory(session_id, user_input, detail_limit))

    if include_long_term:
        try:
            detailed.extend(
                recall_long_term_memory(
                    user_input,
                    limit=detail_limit,
                    threshold=cache.conf.get("thresholdrrm"),
                )
            )
        except Exception as exc:
            print(f"[WARNING] Long-term memory recall failed: {exc}")

    # Global cap after merging conversation details and long-term memory.
    detailed.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            _parse_dt(item.get("date") or item.get("last_seen")).timestamp(),
        ),
        reverse=True,
    )
    detailed = detailed[:max(1, int(detail_limit))]

    return summaries, detailed


def format_memory_context(summaries: list[dict[str, Any]],detailed: list[dict[str, Any]],) -> str:
    parts: list[str] = []

    if summaries:
        lines = ["Summary memory:"]
        for index, item in enumerate(summaries, start=1):
            lines.append(
                f"[{index}] topic={item['topic']} "
                f"date={item['last_seen']} "
                f"similarity={item['similarity']:.4f}\n"
                f"Summary: {item['content']}"
            )
        parts.append("\n\n".join(lines))

    if detailed:
        lines = ["Detailed recalled memory:"]
        for index, item in enumerate(detailed, start=1):
            content = str(item.get("content", "")).strip()
            if not content:
                continue

            role = item.get("role", "system")
            date = item.get("date") or item.get("last_seen", "")
            score = float(item.get("score", 0.0))
            lines.append(
                f"[{index}] role={role} date={date} score={score:.4f}\n"
                f"Evidence: {content}"
            )
        parts.append("\n\n".join(lines))

    return "\n\n".join(parts)
