from __future__ import annotations

from .state import *
from datetime import date
from .memory import add_to_faiss, generate_embedding, recall_from_faiss
from .promptmanager import get_prompt_manager
from .addonsfunction import compute_importance
from .memorystore import recall_long_term_memory, _safe_fts_query, _lexical_score


DEFAULT_ACTIVE_TOPIC_THRESHOLD = cache.conf.get('ActTopicThr')
DEFAULT_SUMMARY_RECALL_THRESHOLD = cache.conf.get('SumReclThr')
DEFAULT_ACTIVE_MEMORY_TOKEN_LIMIT = cache.conf.get('ActMemTokLim')
DEFAULT_DETAIL_RECALL_LIMIT = cache.conf.get('DetReclLim')
DEFAULT_SUMMARY_RECALL_LIMIT = cache.conf.get('SumRecLim')



def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _active_scan_limit() -> int:
    return max(8, _safe_int(_conf("activeMemoryScanLimit", 48), 48))


def _archive_batch_limit() -> int:
    return max(1, _safe_int(_conf("activeMemoryArchiveBatch", 12), 12))


def _safe_rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    try:
        return reranka(query=query, candidates=candidates, top_k=top_k, text_key="content")
    except Exception as exc:
        print(f"[WARNING] Reranker unavailable; using retrieval score: {exc}")
        return sorted(
            candidates,
            key=lambda item: float(item.get("semantic_score", item.get("score", 0.0))),
            reverse=True,
        )[:max(1, int(top_k))]


def _archive_rows_safely(session_id: int, rows: list[tuple]) -> bool:
    """Archive only after a durable summary exists. Never destroy active rows on summary failure."""
    if not rows:
        return False
    try:
        result = _summarize_archived_turns(session_id, rows)
        if not isinstance(result, dict) or not result.get("success"):
            print("[WARNING] Archive skipped; summary was not created.")
            return False
        return True
    except Exception as exc:
        print(f"[WARNING] Archive transaction failed; active rows kept: {exc}")
        return False


def _fit_active_rows(rows: list[tuple], token_limit: int) -> tuple[list[tuple], list[tuple]]:
    manager = get_prompt_manager()
    kept = list(rows)
    removed: list[tuple] = []
    while len(kept) > 1:
        try:
            tokens = manager.estimate_tokens(_turn_messages(kept))
        except Exception as exc:
            print(f"[WARNING] Active token estimation failed: {exc}")
            break
        if tokens <= token_limit:
            break
        removed.append(kept.pop(0))
    return kept, removed


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
    if getattr(cache, "_conversation_memory_ready", False):
        return
    conn = sqlite3.connect(cache.DB_FILE, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS active_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'trimmed', 'archived'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_active_memory_session_status
            ON active_memory(session_id, status, created_at)
        """)
        conn.execute("""
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
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_summaries_session
            ON conversation_summaries(session_id, last_seen)
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS conversation_summaries_fts USING fts5(
                summary_id UNINDEXED,
                topic,
                summary,
                last_seen UNINDEXED
            )
        """)
        # Rebuild only missing FTS rows; SQL tables remain authoritative.
        missing = conn.execute(
            """
            SELECT s.id, s.topic, s.summary, s.last_seen
            FROM conversation_summaries AS s
            LEFT JOIN conversation_summaries_fts AS f ON f.rowid = s.id
            WHERE f.rowid IS NULL
            """
        ).fetchall()
        for summary_id, topic, summary, last_seen in missing:
            conn.execute(
                """
                INSERT INTO conversation_summaries_fts
                    (rowid, summary_id, topic, summary, last_seen)
                VALUES (?, ?, ?, ?, ?)
                """,
                (summary_id, str(summary_id), topic, summary, last_seen),
            )
        conn.execute(
            """
            DELETE FROM conversation_summaries_fts
            WHERE rowid NOT IN (SELECT id FROM conversation_summaries)
            """
        )
        conn.commit()
        cache._conversation_memory_ready = True
    except Exception as exc:
        conn.rollback()
        cache._conversation_memory_ready = False
        print(f"[WARNING] Conversation-memory initialization failed: {exc}")
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
        rows = _active_rows(session_id, limit=max_turns)
    except Exception:
        rows = []

    recent_texts = [row[1] for row in rows[-max_turns:] if row[1]]
    context = " ".join(recent_texts)[-max_chars:]

    if not context:
        return user_input

    return f"{context} {user_input}".strip()

def _active_rows(session_id: int, limit: int | None = None) -> list[tuple]:
    """Return the newest active turns in chronological order.

    The optional limit prevents a long-running session from loading every active
    row into Python just to build a small working-memory window.
    """
    init_conversation_memory()
    conn = sqlite3.connect(cache.DB_FILE, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        if limit is None:
            return conn.execute(
                """
                SELECT id, user_content, assistant_content, created_at
                FROM active_memory
                WHERE session_id = ? AND status = 'active'
                ORDER BY datetime(created_at) ASC, id ASC
                """,
                (session_id,),
            ).fetchall()

        limit = max(1, int(limit))
        rows = conn.execute(
            """
            SELECT id, user_content, assistant_content, created_at
            FROM active_memory
            WHERE session_id = ? AND status = 'active'
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        rows.reverse()
        return rows
    finally:
        conn.close()


def _oldest_active_rows(session_id: int, limit: int) -> list[tuple]:
    init_conversation_memory()
    conn = sqlite3.connect(cache.DB_FILE, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn.execute(
            """
            SELECT id, user_content, assistant_content, created_at
            FROM active_memory
            WHERE session_id = ? AND status = 'active'
            ORDER BY datetime(created_at) ASC, id ASC
            LIMIT ?
            """,
            (session_id, max(1, int(limit))),
        ).fetchall()
    finally:
        conn.close()


def _active_count(session_id: int) -> int:
    init_conversation_memory()
    conn = sqlite3.connect(cache.DB_FILE, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        row = conn.execute(
            "SELECT COUNT(*) FROM active_memory WHERE session_id = ? AND status = 'active'",
            (session_id,),
        ).fetchone()
        return int(row[0] or 0)
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


def _summarize_archived_turns(session_id: int,rows: list[tuple],) -> dict[str, Any] | None:
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

    try:
        completion = cache.client.chat.completions.create(
            model=cache.Sum_model,
            messages=[
                {"role": "system", "content": "Maintain concise persistent conversation memory."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=_safe_int(_conf("comMaxSumToken", 500), 500),
        )
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return {"success": False, "error": "SUM model returned no choices."}
        raw = (getattr(choices[0].message, "content", None) or "").strip()
    except Exception as exc:
        return {"success": False, "error": f"SUM model failure: {exc}"}

    if not raw:
        return {"success": False, "error": "SUM model returned an empty summary."}

    topic = "general"
    summary = raw
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("TOPIC:"):
            topic = line.split(":", 1)[1].strip() or "general"
        elif line.upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    if not summary:
        return {"success": False, "error": "Parsed summary is empty."}

    first_seen = min((_parse_dt(row[3]) for row in rows), default=datetime.now(timezone.utc)).isoformat()
    last_seen = max((_parse_dt(row[3]) for row in rows), default=datetime.now(timezone.utc)).isoformat()
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        init_conversation_memory()
        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            cursor = conn.execute(
                """
                INSERT INTO conversation_summaries (
                    session_id, topic, summary, source_count,
                    first_seen, last_seen, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, topic, summary, len(rows), first_seen, last_seen, created_at),
            )
            summary_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO conversation_summaries_fts
                    (rowid, summary_id, topic, summary, last_seen)
                VALUES (?, ?, ?, ?, ?)
                """,
                (summary_id, str(summary_id), topic, summary, last_seen),
            )

            row_ids = [int(row[0]) for row in rows]
            placeholders = ",".join("?" for _ in row_ids)
            conn.execute(
                f"UPDATE active_memory SET status = 'archived' WHERE id IN ({placeholders}) AND status = 'active'",
                row_ids,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    except Exception as exc:
        return {"success": False, "error": f"Summary/archive database failure: {exc}"}

    # Semantic indexing is optional. The SQL summary is already durable.
    try:
        if getattr(cache, "faiss_index", None) is not None:
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
        print(f"[WARNING] Summary semantic index failed; SQL summary kept: {exc}")

    return {
        "success": True,
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
    """Prepare working memory without ever deleting/archiving on a transient error."""
    try:
        init_conversation_memory()
        scan_limit = _active_scan_limit()
        archive_limit = _archive_batch_limit()
        active_count = _active_count(session_id)
        if active_count > scan_limit:
            stale_count = min(archive_limit, active_count - scan_limit)
            stale_rows = _oldest_active_rows(session_id, stale_count)
            _archive_rows_safely(session_id, stale_rows)
        rows = _active_rows(session_id, limit=scan_limit)
    except Exception as exc:
        print(f"[WARNING] Active-memory read failed: {exc}")
        return []

    if not rows:
        return []

    # Keep the working set bounded. Archive pressure is handled before the
    # embedding scan so a long session does not turn every turn into O(N).
    topic_threshold = _active_topic_threshold()
    if _is_short_continuation(user_input):
        topic_threshold = min(topic_threshold, 0.25)

    try:
        query_vector = np.asarray(
            generate_embedding(user_input, embedding_type="query"),
            dtype="float32",
        ).reshape(1, -1)
    except Exception as exc:
        print(f"[WARNING] Active-memory embedding failed; using recent turns: {exc}")
        recent = rows[-max(1, min(6, len(rows))):]
        kept, dropped = _fit_active_rows(recent, _active_token_limit())
        if dropped:
            _archive_rows_safely(session_id, dropped)
        return _turn_messages(kept)

    related_rows: list[tuple] = []
    archive_candidates: list[tuple] = []
    query_vec = _normalize(query_vector[0])

    for row in rows:
        try:
            row_id, user_content, assistant_content, _ = row
            turn_text = f"USER: {user_content}\nASSISTANT: {assistant_content}"
            vector = np.asarray(
                generate_embedding(turn_text, embedding_type="passage"),
                dtype="float32",
            ).reshape(1, -1)
            score = float(np.dot(query_vec, _normalize(vector[0])))
        except Exception as exc:
            print(f"[WARNING] Active-memory row {row[0] if row else '?'} could not be embedded: {exc}")
            # Keep uncertain rows active; do not archive data because of one bad vector.
            related_rows.append(row)
            continue

        if score >= topic_threshold:
            row = tuple(row) + (score,) if len(row) == 4 else row
            related_rows.append(row)
        else:
            archive_candidates.append(row)

        print(
            f"[ACTIVE MEMORY] id={row[0]} topic_score={score:.4f} "
            f"threshold={topic_threshold:.4f} "
            f"status={'keep' if score >= topic_threshold else 'archive-candidate'}"
        )

    # Archive is transactional: failed summarization leaves rows active.
    if archive_candidates:
        _archive_rows_safely(session_id, archive_candidates[:_archive_batch_limit()])

    # Strip temporary score fields and rerank the candidates.
    candidates = []
    for row in related_rows:
        score = float(row[4]) if len(row) > 4 else 0.0
        candidates.append({
            "content": f"USER: {row[1]}\nASSISTANT: {row[2]}",
            "row": row[:4],
            "semantic_score": score,
        })

    reranked = _safe_rerank(user_input, candidates, top_k=min(8, len(candidates)))
    ordered_rows = [item["row"] for item in reranked if item.get("row")]

    try:
        manager = get_prompt_manager()
        token_limit = _active_token_limit()
        kept, dropped = _fit_active_rows(ordered_rows, token_limit)
    except Exception as exc:
        print(f"[WARNING] Active-memory budget handling failed: {exc}")
        kept, dropped = ordered_rows[-4:], ordered_rows[:-4]

    if dropped:
        # Dropped-for-budget rows are safely archived when possible. A failed
        # summary keeps them active, so no runtime error becomes data loss.
        _archive_rows_safely(session_id, dropped)

    return _turn_messages(kept)



def record_active_turn(session_id: int,user_content: str,assistant_content: str,created_at: str | None = None,) -> int:
    init_conversation_memory()
    timestamp = _parse_dt(created_at).isoformat() if created_at else datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(cache.DB_FILE, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        cursor = conn.execute(
            """
            INSERT INTO active_memory (
                session_id, user_content, assistant_content, created_at, status
            ) VALUES (?, ?, ?, ?, 'active')
            """,
            (session_id, str(user_content), str(assistant_content), timestamp),
        )
        row_id = int(cursor.lastrowid)
        conn.commit()
    except Exception as exc:
        print(f"[WARNING] Active-memory DB write failed: {exc}")
        return -1
    finally:
        conn.close()

    try:
        if getattr(cache, "faiss_index", None) is not None:
            add_to_faiss(
                cache.faiss_index,
                cache.id_map,
                f"USER: {user_content}\nASSISTANT: {assistant_content}",
                f"active:{row_id}",
                meta_type="active",
                importance=compute_importance(str(user_content) + " " + str(assistant_content)),
                session_id=session_id,
                memory_id=row_id,
            )
    except Exception as exc:
        print(f"[WARNING] Failed to index active memory; DB row kept: {exc}")

    return row_id



def _summary_hits(session_id: int,user_input: str,limit: int,threshold: float | None = None,) -> list[dict[str, Any]]:
    """Hybrid summary retrieval: FTS guarantees lexical recall; FAISS adds semantic recall."""
    try:
        init_conversation_memory()
        limit = max(1, int(limit))
        query_text = _recall_query_text(session_id, user_input)
        fts_query = _safe_fts_query(query_text)
        date_hint = _parse_date_hint(user_input)
        candidate_limit = max(limit * 8, 40)
        candidate_map: dict[int, dict[str, Any]] = {}

        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            rows = []
            if fts_query:
                try:
                    if date_hint:
                        rows = conn.execute(
                            """
                            SELECT s.id, s.topic, s.summary, s.first_seen,
                                   s.last_seen, s.source_count,
                                   bm25(conversation_summaries_fts) AS bm25_score
                            FROM conversation_summaries_fts AS f
                            JOIN conversation_summaries AS s ON s.id = f.rowid
                            WHERE conversation_summaries_fts MATCH ?
                              AND date(s.first_seen) <= date(?)
                              AND date(s.last_seen) >= date(?)
                            ORDER BY bm25_score ASC, datetime(s.last_seen) DESC
                            LIMIT ?
                            """,
                            (fts_query, date_hint.isoformat(), date_hint.isoformat(), candidate_limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT s.id, s.topic, s.summary, s.first_seen,
                                   s.last_seen, s.source_count,
                                   bm25(conversation_summaries_fts) AS bm25_score
                            FROM conversation_summaries_fts AS f
                            JOIN conversation_summaries AS s ON s.id = f.rowid
                            WHERE conversation_summaries_fts MATCH ?
                            ORDER BY bm25_score ASC, datetime(s.last_seen) DESC
                            LIMIT ?
                            """,
                            (fts_query, candidate_limit),
                        ).fetchall()
                except sqlite3.OperationalError as exc:
                    print(f"[WARNING] Summary FTS retrieval failed: {exc}")
                    rows = []

            bm25_values = [float(row[6]) for row in rows] if rows else []
            best_bm25 = min(bm25_values) if bm25_values else 0.0
            worst_bm25 = max(bm25_values) if bm25_values else 0.0
            span = worst_bm25 - best_bm25

            for row in rows:
                summary_id, topic, summary, first_seen, last_seen, source_count, bm25_score = row
                lexical = _lexical_score(query_text, f"{topic} {summary}")
                bm25_norm = 1.0 if span <= 1e-12 else (worst_bm25 - float(bm25_score)) / span
                item = {
                    "summary_id": int(summary_id),
                    "topic": topic,
                    "content": summary,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "source_count": int(source_count),
                    "lexical_score": float(max(0.0, min(1.0, lexical * 0.8 + bm25_norm * 0.2))),
                    "similarity": 0.0,
                }
                candidate_map[int(summary_id)] = item

            # Semantic candidates are optional and may be stale; validate them against SQL.
            try:
                vector = np.asarray(
                    generate_embedding(query_text, embedding_type="query"),
                    dtype="float32",
                ).reshape(1, -1)
                semantic_hits = recall_from_faiss(
                    vector,
                    topk=candidate_limit,
                    threshold=0.0,
                    meta_type="sum",
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
                        SELECT id, topic, summary, first_seen, last_seen, source_count
                        FROM conversation_summaries
                        WHERE id IN ({placeholders})
                        """,
                        semantic_ids,
                    ).fetchall()
                    for row in extra_rows:
                        sid = int(row[0])
                        if date_hint and not (
                            _parse_dt(row[3]).date() <= date_hint <= _parse_dt(row[4]).date()
                        ):
                            continue
                        candidate_map.setdefault(
                            sid,
                            {
                                "summary_id": sid,
                                "topic": row[1],
                                "content": row[2],
                                "first_seen": row[3],
                                "last_seen": row[4],
                                "source_count": int(row[5]),
                                "lexical_score": _lexical_score(query_text, f"{row[1]} {row[2]}"),
                                "similarity": 0.0,
                            },
                        )
                for hit in semantic_hits:
                    memory_id = hit.get("memory_id")
                    if memory_id is not None and int(memory_id) in candidate_map:
                        candidate_map[int(memory_id)]["similarity"] = max(
                            candidate_map[int(memory_id)]["similarity"],
                            float(hit.get("similarity", 0.0)),
                        )
            except Exception as exc:
                print(f"[WARNING] Summary semantic retrieval unavailable: {exc}")

        finally:
            conn.close()

        ranked = []
        semantic_available = any(item.get("similarity", 0.0) > 0 for item in candidate_map.values())
        for item in candidate_map.values():
            recency = 1.0 / (1.0 + max(0.0, (time.time() - _parse_dt(item["last_seen"]).timestamp()) / 86400.0) / 30.0)
            source_score = min(1.0, item["source_count"] / 5.0)
            if date_hint:
                date_score = 1.0 if _date_distance(item, date_hint) == 0 else 0.0
            else:
                date_score = 0.0

            if semantic_available:
                score = item["similarity"] * 0.65 + item["lexical_score"] * 0.25 + source_score * 0.05 + recency * 0.05
            else:
                score = item["lexical_score"] * 0.70 + source_score * 0.10 + recency * 0.20
            if date_hint:
                score = date_score * 0.70 + score * 0.30
            item["score"] = float(max(0.0, min(1.0, score)))
            ranked.append(item)

        if threshold is not None:
            try:
                value = float(threshold)
                ranked = [item for item in ranked if item["score"] >= value]
            except (TypeError, ValueError):
                pass

        ranked.sort(key=lambda item: (item["score"], _parse_dt(item["last_seen"]).timestamp()), reverse=True)
        return ranked[:limit]
    except Exception as exc:
        print(f"[WARNING] Summary recall failed: {exc}")
        return []



def has_summary_memory(session_id: int, user_input: str) -> bool:
    """Return whether the current input matches an archived topic summary."""
    return bool(_summary_hits(session_id, user_input, limit=1))


def recall_detailed_memory(session_id: int,user_input: str,limit: int = DEFAULT_DETAIL_RECALL_LIMIT,) -> list[dict[str, Any]]:
    try:
        query = _safe_fts_query(user_input)
        if not query:
            return []

        limit = max(1, int(limit))
        candidate_limit = max(limit * 8, 40)
        target_date = _parse_date_hint(user_input)

        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            params: list[Any] = [query]
            date_clause = ""
            if target_date:
                date_clause = "AND date(created_at) = ?"
                params.append(target_date.isoformat())
            params.append(candidate_limit)
            try:
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
            except sqlite3.OperationalError as exc:
                print(f"[WARNING] Detailed FTS retrieval failed: {exc}")
                rows = []
        finally:
            conn.close()

        current_text = " ".join(str(user_input).split()).casefold()
        rows = [r for r in rows if " ".join(str(r[3]).split()).casefold() != current_text]
        if not rows and target_date:
            return []
        if not rows:
            return []

        bm25_values = [float(row[5]) for row in rows]
        best = min(bm25_values)
        worst = max(bm25_values)
        span = worst - best

        candidates = []
        base_by_id = {}
        now_ts = time.time()
        for row in rows:
            rowid, row_session_id, role, content, created_at, bm25_score = row
            content = str(content)
            if len(content) > 1000:
                content = content[:1000]
            lexical = _lexical_score(user_input, content)
            fts_score = 1.0 if span <= 1e-12 else (worst - float(bm25_score)) / span
            age_days = max(0.0, (now_ts - _parse_dt(created_at).timestamp()) / 86400.0)
            recency = 1.0 / (1.0 + age_days)
            current_session = 1.0 if row_session_id == session_id else 0.0
            item = {
                "role": role,
                "content": content,
                "date": created_at,
                "session_id": row_session_id,
                "fts_score": float(max(0.0, min(1.0, 0.75 * lexical + 0.25 * fts_score))),
                "recency_score": recency,
                "session_score": current_session,
                "score": 0.0,
                "rowid": int(rowid),
            }
            candidates.append({
                "content": content,
                "row": item,
            })
            base_by_id[int(rowid)] = item

        reranked = _safe_rerank(user_input, candidates, top_k=min(candidate_limit, len(candidates)))
        results = []
        for item in reranked:
            base = item.get("row")
            if not isinstance(base, dict):
                continue
            rerank_score = float(item.get("rerank_score", base.get("fts_score", 0.0)))
            if target_date:
                date_score = 1.0 if _parse_dt(base["date"]).date() == target_date else 0.0
                score = date_score * 0.75 + rerank_score * 0.25
            else:
                score = (
                    rerank_score * 0.65
                    + base["fts_score"] * 0.15
                    + base["recency_score"] * 0.10
                    + base["session_score"] * 0.10
                )
            base["rerank_score"] = rerank_score
            base["score"] = float(max(0.0, min(1.0, score)))
            results.append(base)

        results.sort(
            key=lambda item: (item["score"], _parse_dt(item["date"]).timestamp()),
            reverse=True,
        )
        return results[:limit]
    except Exception as exc:
        print(f"[WARNING] Detailed memory recall failed: {exc}")
        return []



def recall_memory_context(session_id: int,user_input: str,summary_limit: int = DEFAULT_SUMMARY_RECALL_LIMIT,detail_limit: int = DEFAULT_DETAIL_RECALL_LIMIT,include_long_term: bool = True,) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Gather summaries, conversation evidence, and long-term facts independently."""
    try:
        summaries = _summary_hits(session_id, user_input, summary_limit)
    except Exception as exc:
        print(f"[WARNING] Summary stage failed: {exc}")
        summaries = []

    try:
        conversation = recall_detailed_memory(session_id, user_input, detail_limit)
    except Exception as exc:
        print(f"[WARNING] Detail stage failed: {exc}")
        conversation = []

    long_term: list[dict[str, Any]] = []
    if include_long_term:
        try:
            long_term = recall_long_term_memory(
                user_input,
                limit=detail_limit,
                threshold=cache.conf.get("thresholdrrm"),
            )
        except Exception as exc:
            print(f"[WARNING] Long-term memory recall failed: {exc}")

    # De-duplicate by normalized text while preserving the strongest result.
    unique_long_term: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in long_term:
        key = " ".join(str(item.get("content", "")).split()).casefold()
        if key and key not in seen:
            seen.add(key)
            unique_long_term.append(item)
    long_term = unique_long_term

    detail_limit = max(1, int(detail_limit))
    target_date = _parse_date_hint(user_input)

    if target_date:
        # Exact-date conversational evidence gets first priority. Long-term
        # memories are only used as additional evidence, not as a replacement.
        date_conversation = [
            item for item in conversation
            if _parse_dt(item.get("date")).date() == target_date
        ]
        date_long_term = [
            item for item in long_term
            if _parse_dt(item.get("date") or item.get("last_seen")).date() == target_date
        ]
        detailed = (date_conversation + date_long_term)[:detail_limit]
        if len(detailed) < detail_limit:
            remaining_pool = [
                item for item in conversation + long_term
                if item not in detailed and
                " ".join(str(item.get("content", "")).split()).casefold() not in {
                    " ".join(str(x.get("content", "")).split()).casefold() for x in detailed
                }
            ]
            detailed.extend(remaining_pool[: detail_limit - len(detailed)])
    else:
        # Reserve room for both persistent facts and actual conversational evidence.
        lt_quota = min(len(long_term), max(1, (detail_limit + 1) // 2))
        conv_quota = min(len(conversation), detail_limit - lt_quota)
        if conv_quota == 0 and len(conversation) > 0:
            conv_quota = min(len(conversation), detail_limit)
            lt_quota = min(len(long_term), detail_limit - conv_quota)
        detailed = long_term[:lt_quota]
        detailed.extend(conversation[:conv_quota])
        if len(detailed) < detail_limit:
            pool = long_term[lt_quota:] + conversation[conv_quota:]
            seen_detail = {
                " ".join(str(item.get("content", "")).split()).casefold()
                for item in detailed
            }
            for item in pool:
                key = " ".join(str(item.get("content", "")).split()).casefold()
                if key in seen_detail:
                    continue
                detailed.append(item)
                seen_detail.add(key)
                if len(detailed) >= detail_limit:
                    break

    return summaries, detailed[:detail_limit]



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
