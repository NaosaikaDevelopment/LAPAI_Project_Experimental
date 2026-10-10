from ...statecore import *


def memory_commit(
    kind: Literal[
        "question", "fact", "preference", "identity", "goal",
        "project", "event", "instruction", "small_talk"],
    category: str, confidence: float = DEFAULT_MEMORY_CONFIDENCE,) -> dict[str, Any]:
    """Classify the current user input and update memory state.

    @required_each_turn
    @provides_state: memory_query

    Label the current user message. The system already stores the message
    text, so do NOT copy the message or any explanation into the arguments.

    Args:
        kind: Exactly one allowed value. Use "fact" for statements about the user's life
            and "question" whenever the user asks something.
        category: One or two lowercase words, e.g. "preference", "food", "education".
            Never a sentence.
        confidence:  A separate number from 0.0 to 1.0, e.g. 0.9.
            Never write it inside category.
    """
    user_text = getattr(cache, "current_user_msg", "")
    return finalize_memory_commit(user_text, kind, category, confidence)

