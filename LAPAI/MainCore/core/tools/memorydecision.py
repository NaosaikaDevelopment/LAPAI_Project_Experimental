from ...statecore import *
def recallmemory() -> str:
    """
    Recall relevant long-term memory for the current user query.

    @requires_state: memory_query
    @required_when_state: memory_query
    The retrieved memory is returned directly to the model through the
    tool result.
    """

    user_msg = getattr(cache, "current_user_msg", "")
    limit = cache.conf.get('limitrrm') or 5

    recalled = recall_relevant_memory(
        user_msg,
        limit=limit
    )

    if not recalled:
        return "No relevant memory was found."

    lines = ["Relevant long-term memory:"]

    for index, item in enumerate(recalled, start=1):
        content = str(item.get("content", "")).strip()

        if not content:
            continue

        parts = []

        for field, label in (
            ("kind", "kind"),
            ("category", "category"),
            ("created_at", "date"),
            ("timestamp", "date"),
            ("relation_score", "relation"),
            ("fts_score", "fts"),
            ("faiss_score", "faiss"),
            ("confidence", "confidence"),
            ("support_count", "support"),
            ("score", "score"),
        ):
            value = item.get(field)

            if value is not None:
                parts.append(f"{label}={value}")

        metadata = " | ".join(parts)

        if metadata:
            lines.append(
                f"[{index}] {metadata}\n"
                f"Evidence: {content}"
            )
        else:
            lines.append(
                f"[{index}] Evidence: {content}"
            )

    return "\n\n".join(lines)