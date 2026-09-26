from ..state import *
# Testing tools ~experimental~ it is now work perfectly
def get_time() -> str:
    """
    Get the current system date and time.
    """
    return datetime.now().astimezone().isoformat()

def add(a: int, b: int):
    """
    Add two numbers.
    """
    return a + b