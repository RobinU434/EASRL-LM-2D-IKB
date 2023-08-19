import logging
import os
from typing import Any, Callable

import psutil


def profile_runtime(func: Callable[..., Any]) -> Callable[..., Any]:
    """decorator to monitor runtime of a given function

    Args:
        func (Callable[..., Any]): function to wrap in runtime monitor

    Returns:
        Callable[..., Any]: function wrapped around the given function
    """

    def wrapper(*args, **kwargs):
        import time

        logging.info(f"started {func.__name__}")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time = time.perf_counter() - start
        logging.info(f"needed {time:.4f} ms to run {func.__name__}")

        return result

    return wrapper


# inner psutil function
def _process_memory() -> int:
    """utility function for profile memory

    Returns:
        int: memory consumption
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


# decorator function
def profile_memory(func: Callable[..., Any]) -> Callable[..., Any]:
    """decorator function to get memory consumption

    Args:
        func (Callable[..., Any]): function to call

    Returns:
        Callable[..., Any]: function to call wrapped in a memory consumption function
    """

    def wrapper(*args, **kwargs):
        mem_before = _process_memory()
        result = func(*args, **kwargs)
        mem_after = _process_memory()
        logging.info(
            "{}:consumed memory: {:,}".format(
                func.__name__, mem_before, mem_after, mem_after - mem_before
            )
        )

        return result

    return wrapper


def not_implemented_warning(func: Callable[..., Any]) -> Callable[..., Any]:
    """decorator for not implemented warning

    Args:
        func (Callable[..., Any]): function to call

    Returns:
        Callable[..., Any]: function to call wrapped in logging.warning method
    """

    def wrapper(*args, **kwargs):
        logging.warning(f"Function {func.__name__} is not implemented")
        func(*args, **kwargs)

    return wrapper
