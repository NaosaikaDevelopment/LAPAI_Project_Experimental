from ...statecore import *

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
def turnleda() -> str:
    """
    To turn LED A on
    """
    cache.conf['conditionLEDA'] = True
