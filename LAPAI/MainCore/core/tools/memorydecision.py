from ...statecore import *

def recallmemory() -> str:
    """Recall historical memory; retrieval errors degrade to an empty result.

    @requires_state: memory_query
    @required_when_state: memory_query
    """

    try:
        return recall_memory_text()
    except Exception as exc:
        print(f"[WARNING] Memory recall tool failed: {exc}")
        return "No relevant historical memory was found."

