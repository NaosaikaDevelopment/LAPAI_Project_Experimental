from ...statecore import *


def memory_commit(
kind: Literal[
    "question", "fact", "preference", "identity", "goal",
    "project", "event", "instruction", "small_talk"
],
category: str,
confidence: float,) -> dict[str, Any]:
    """Classify the current user input and update memory state.

    @required_each_turn
    @provides_state: memory_query

    kind must be one of the allowed classification values.
    category is a short logical topic label.
    confidence is from 0.0 to 1.0.
    """
    user_text = getattr(cache, "current_user_msg", "")

    raw_kind = str(kind).strip().casefold()
    raw_category = str(category).strip().casefold()
    recovered_kind = None
    recovered_category = None
    if raw_kind not in VALID_MEMORY_KINDS:
        matches = [
            (raw_kind.find(candidate), candidate)
            for candidate in VALID_MEMORY_KINDS
            if raw_kind.find(candidate) >= 0
        ]
        if matches:
            recovered_kind = min(matches, key=lambda item: item[0])[1]
            kind = recovered_kind
        else:
            return {
                "success": False,
                "stored": False,
                "error": f"Invalid memory kind: {raw_kind}",
            }

    # Categories are intentionally open-ended, but cap obvious documentation
    # echoes so they do not become persistent junk labels.
    if (len(raw_category) > 80 or "\n" in raw_category or "\r" in raw_category):
        user_lower = str(user_text).casefold()
        if any(token in user_lower for token in ("fav", "favorite", "favourite", "prefer", "preference")):
            recovered_category = "preference"
        elif any(token in user_lower for token in ("remember", "memory", "recall")):
            recovered_category = "memory"
        else:
            category_hints = (
                "preference", "identity", "goal", "project", "fact",
                "event", "education", "food", "memory", "personal", "general"
            )
            matches = [
                (raw_category.find(hint), hint)
                for hint in category_hints
                if raw_category.find(hint) >= 0
            ]
            if matches:
                recovered_category = min(matches, key=lambda item: item[0])[1]

        category = recovered_category or "general"
    else:
        category = raw_category or "general"

    result = commit_memory(
        session_id=cache.session_id,
        user_text=user_text,
        kind=kind,
        category=category,
        confidence=confidence,
        created_at=cache.crntTime,
    )

    # Tool-owned state. The core only transports this state generically.
    normalized_category = category.strip().casefold()
    legacy_memory_query = (
        kind.strip().casefold() == "question"
        and (
            normalized_category in {
                "preference",
                "identity",
                "goal",
                "project",
                "fact",
                "event",
                "personal",
                "memory",
            }
            or any(
                hint in normalized_category
                for hint in (
                    "favorite",
                    "favourite",
                    "prefer",
                    "profile",
                    "personal",
                    "memory",
                    "user_",
                )
            )
        )
    )

    try:
        summary_match = has_summary_memory(
            cache.session_id,
            user_text,
        )
    except Exception as exc:
        summary_match = False
        result["recall_warning"] = str(exc)

    memory_query = bool(summary_match or legacy_memory_query)

    result["_state"] = {
        "memory_query": memory_query
    }

    return result
