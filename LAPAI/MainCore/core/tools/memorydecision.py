from ...statecore import *

def recallmemory() -> str:
    """
    Recall historical memory using a two-stage pipeline.

    @requires_state: memory_query
    @required_when_state: memory_query

    Stage 1 retrieves topic summaries (`sum` memory). If a relevant summary
    exists, stage 2 retrieves detailed conversation evidence ranked by an
    explicit date when present, otherwise by semantic relevance and recency.
    """

    user_msg = getattr(cache, "current_user_msg", "")
    summary_limit = cache.conf.get("summaryRecallLimit") or 3
    detail_limit = cache.conf.get("detailRecallLimit") or 8

    summaries, detailed = recall_memory_context(
        session_id=cache.session_id,
        user_input=user_msg,
        summary_limit=summary_limit,
        detail_limit=detail_limit,
        include_long_term=True,
    )

    formatted = format_memory_context(summaries, detailed)
    if not formatted:
        return "No relevant historical memory was found."

    return formatted
