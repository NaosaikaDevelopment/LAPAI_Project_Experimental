from .state import *
def compact_old_memory(client,model,session_id,minutes,max_summary_tokens=cache.conf.get('comMaxSumToken'),):
    """Time-based compaction with transactional progress updates.

    The summarized cursor is advanced only after a non-empty summary is
    successfully generated and persisted. A SUM-model failure therefore never
    causes old messages to become permanently skipped.
    """
    try:
        state = get_memory_state(session_id)
        previous_summary = state["summary"]
        summarized_until = state["summarized_until"] or "1970-01-01 00:00:00"
        summarized_until_rowid = state["summarized_until_rowid"]

        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            if summarized_until_rowid > 0:
                rows = conn.execute(
                    """
                    SELECT rowid, role, content, created_at
                    FROM messages
                    WHERE rowid > ?
                      AND session_id = ?
                      AND role != 'summary'
                      AND datetime(created_at) <= datetime('now', ?)
                    ORDER BY rowid ASC
                    """,
                    (summarized_until_rowid, session_id, f"-{minutes} minutes"),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT rowid, role, content, created_at
                    FROM messages
                    WHERE session_id = ?
                      AND role != 'summary'
                      AND datetime(created_at) > datetime(?)
                      AND datetime(created_at) <= datetime('now', ?)
                    ORDER BY rowid ASC
                    """,
                    (session_id, summarized_until, f"-{minutes} minutes"),
                ).fetchall()
        finally:
            conn.close()

        if not rows:
            return None

        expired_text = "\n".join(
            f"{str(role).upper()}: {content}"
            for _, role, content, _ in rows
        )
        prompt_text = f"""
You are maintaining persistent memory for a conversation.

Previous summary:
{previous_summary or '(none)'}

New conversation segment:
{expired_text}

Update the persistent summary.

Keep important user information, ongoing projects, decisions, technical
specifics, unresolved problems, important context, and relevant preferences.
Remove greetings, filler, repetition, and temporary noise.

Return only the updated summary.
"""

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Maintain a concise persistent conversation memory."},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=max_summary_tokens,
            )
            choices = getattr(completion, "choices", None) or []
            new_summary = (
                getattr(choices[0].message, "content", "").strip()
                if choices else ""
            )
        except Exception as exc:
            print(f"[WARNING] Memory compaction model failed; cursor unchanged: {exc}")
            return None

        if not new_summary:
            print("[WARNING] Memory compaction returned an empty summary; cursor unchanged.")
            return None

        latest_rowid = int(rows[-1][0])
        latest_timestamp = rows[-1][3]
        try:
            save_memory_state(
                session_id,
                new_summary,
                latest_timestamp,
                latest_rowid,
            )
        except Exception as exc:
            print(f"[WARNING] Memory compaction state save failed: {exc}")
            return None

        return new_summary
    except Exception as exc:
        print(f"[WARNING] compact_old_memory failed safely: {exc}")
        return None


def get_memory_state(session_id):
    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT summary, summarized_until, summarized_until_rowid
        FROM memory_state
        WHERE session_id = ?
    """, (session_id,))

    row = c.fetchone()
    conn.close()

    if row is None:
        return {
            "summary": "",
            "summarized_until": None,
            "summarized_until_rowid": 0
        }

    return {
        "summary": row[0] or "",
        "summarized_until": row[1],
        "summarized_until_rowid": int(row[2] or 0)
    }

def save_memory_state(session_id,summary,summarized_until,summarized_until_rowid=0):
    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    c.execute("""
        INSERT INTO memory_state (
            session_id,
            summary,
            summarized_until,
            summarized_until_rowid
        )
        VALUES (?, ?, ?, ?)

        ON CONFLICT(session_id)
        DO UPDATE SET
            summary = excluded.summary,
            summarized_until = excluded.summarized_until,
            summarized_until_rowid = excluded.summarized_until_rowid
    """, (
        session_id,
        summary,
        summarized_until,
        int(summarized_until_rowid or 0)
    ))

    conn.commit()
    conn.close()

def search_memory(query, limit=5):
    """Legacy lexical search kept fail-soft for callers outside the new pipeline."""
    if not query:
        return []
    try:
        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            statement = """
                SELECT rowid, session_id, role, content, created_at,
                       bm25(messages) AS score
                FROM messages
                WHERE messages MATCH ?
                ORDER BY score
                LIMIT ?
            """
            safe_limit = max(1, int(limit))
            try:
                return conn.execute(statement, (query, safe_limit)).fetchall()
            except sqlite3.OperationalError:
                terms = re.findall(r"\w+", str(query), flags=re.UNICODE)
                if not terms:
                    return []
                safe_query = " OR ".join(
                    '\"' + term.replace('\"', '""') + '\"'
                    for term in terms
                )
                return conn.execute(statement, (safe_query, safe_limit)).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        print(f"[WARNING] Legacy search_memory failed: {exc}")
        return []


def recall_recent_memory(session_id, minutes=5, turns=3):
    try:
        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            max_messages = max(1, int(turns)) * 2
            rows = conn.execute("""
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                  AND role NOT IN ('summary', 'system')
                  AND datetime(created_at) >= datetime('now', ?)
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, f"-{minutes} minutes", max_messages)).fetchall()
        finally:
            conn.close()
        rows.reverse()
        return [{"role": role, "content": content, "time": created_at}
                for role, content, created_at in rows]
    except Exception as exc:
        print(f"[WARNING] Legacy recall_recent_memory failed: {exc}")
        return []


def should_recall(user_msg):
    """Legacy SUM gate; failure means recall is not required rather than a crash."""
    try:
        completion = cache.client.chat.completions.create(
            model=cache.Sum_model,
            messages=[{
                "role": "system",
                "content": f"Answer YES or NO. Does this query need past memory to answer?\nQuery: {user_msg}"
            }],
            max_tokens=3
        )
        choices = getattr(completion, "choices", None) or []
        content = getattr(choices[0].message, "content", "") if choices else ""
        return "yes" in str(content).lower()
    except Exception as exc:
        print(f"[WARNING] Legacy should_recall failed: {exc}")
        return False

def summarize_session(client,model,session_id,max_tokens=500): #Old code backward compatible
    """
    Backward-compatible wrapper.
    The old count-based summarization is now time-based compaction.
    """
    return compact_old_memory(
        client=client,
        model=model,
        session_id=session_id,
        minutes=5,
        max_summary_tokens=max_tokens
    )