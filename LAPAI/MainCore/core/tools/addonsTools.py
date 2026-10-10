from ...statecore import *

def get_time() -> str:
    """
    Get the current system date and time.
    """
    return datetime.now().astimezone().isoformat()
