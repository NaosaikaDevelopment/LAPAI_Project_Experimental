from typing import Any

from ..memorystore import commit_memory
from ...statecore import cache


def memory_commit(kind: str, category: str, confidence: float) -> dict[str, Any]:
    """
    Classify the current user input before answering and commit its memory status.

    @required_each_turn

    @provides_state: memory_query

    This tool MUST be called once for every user turn.
    The tool reads the original user input from the current agent context.
    Never rewrite, summarize, or invent the user's text.

    kind:
        question, fact, preference, identity, goal, project, event,
        instruction, or small_talk.

    category:
        Logical topic/category of the input, such as food, identity,
        project, education, preference, or general.

    confidence:
        Classification confidence from 0.0 to 1.0.
    """
    user_text = getattr(cache, "current_user_msg", "")

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
    memory_query = (
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

    result["_state"] = {
        "memory_query": memory_query
    }

    return result
