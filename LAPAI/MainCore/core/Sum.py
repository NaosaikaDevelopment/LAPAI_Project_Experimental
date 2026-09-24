from .state import *
def compact_old_memory(client,model,session_id,minutes,max_summary_tokens=cache.conf.get('comMaxSumToken')):
    state = get_memory_state(session_id)

    previous_summary = state["summary"]
    summarized_until = state["summarized_until"]
    summarized_until_rowid = state["summarized_until_rowid"]

    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    if summarized_until_rowid > 0:
        c.execute("""
            SELECT rowid, role, content, created_at
            FROM messages
            WHERE rowid > ?
              AND session_id = ?
              AND role != 'summary'
              AND datetime(created_at) <= datetime('now', ?)
            ORDER BY rowid ASC
        """, (
            summarized_until_rowid,
            session_id,
            f"-{minutes} minutes"
        ))
    else:
        if summarized_until is None:
            summarized_until = "1970-01-01 00:00:00"

        c.execute("""
            SELECT rowid, role, content, created_at
            FROM messages
            WHERE session_id = ?
              AND role != 'summary'
              AND datetime(created_at) > datetime(?)
              AND datetime(created_at) <= datetime('now', ?)
            ORDER BY rowid ASC
        """, (
            session_id,
            summarized_until,
            f"-{minutes} minutes"
        ))

    rows = c.fetchall()
    conn.close()

    if not rows:
        return None

    expired_text = "\n".join(
        f"{role.upper()}: {content}"
        for _, role, content, _ in rows
    )

    prompt_text = f"""
You are maintaining persistent memory for a conversation.

Previous summary:
{previous_summary or "(none)"}

New conversation segment:
{expired_text}

Update the persistent summary.

Keep:
- important user information
- ongoing projects
- decisions
- technical details
- unresolved problems
- important context
- preferences that matter later

Remove:
- greetings
- filler
- repetition
- temporary conversational noise

Return only the updated summary.
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Maintain a concise persistent conversation memory."
                )
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=max_summary_tokens
    )

    new_summary = (
        completion
        .choices[0]
        .message
        .content
        .strip()
    )

    latest_rowid = int(rows[-1][0])
    latest_timestamp = rows[-1][3]

    save_memory_state(
        session_id,
        new_summary,
        latest_timestamp,
        latest_rowid
    )

    return new_summary

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

def save_memory_state(
    session_id,
    summary,
    summarized_until,
    summarized_until_rowid=0
):
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
    if not query:
        return []

    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    statement = """
        SELECT
            rowid,
            session_id,
            role,
            content,
            created_at,
            bm25(messages) AS score
        FROM messages
        WHERE messages MATCH ?
        ORDER BY score
        LIMIT ?
    """

    try:
        c.execute(statement, (query, limit))
    except sqlite3.OperationalError:
        # SQL parameter binding is safe, but FTS5 has its own query grammar.
        # Retry malformed literal input as an OR query of quoted word tokens.
        terms = re.findall(r"\w+", str(query), flags=re.UNICODE)
        if not terms:
            conn.close()
            return []

        safe_query = " OR ".join(
            '"' + term.replace('"', '""') + '"'
            for term in terms
        )
        c.execute(statement, (safe_query, limit))

    results = c.fetchall()
    conn.close()
    return results


def recall_recent_memory(session_id,minutes=5,turns=3):
    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()

    max_messages = turns * 2

    c.execute("""
        SELECT role, content, created_at
        FROM messages
        WHERE session_id = ?
          AND role NOT IN ('summary', 'system')
          AND datetime(created_at) >= datetime('now', ?)
        ORDER BY created_at DESC
        LIMIT ?
    """, (session_id,f"-{minutes} minutes",max_messages))

    rows = c.fetchall()
    conn.close()

    rows.reverse()

    return [
        {
            "role": role,
            "content": content,
            "time": created_at
        }
        for role, content, created_at in rows
    ]

def should_recall(user_msg):
    completion = cache.client.chat.completions.create(
        model=cache.Sum_model,
        messages=[{
            "role": "system",
            "content": f"Answer YES or NO. Does this query need past memory to answer?\nQuery: {user_msg}"
        }],
        max_tokens=3
    )
    return "yes" in completion.choices[0].message.content.lower()



    
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