"""Module to store various util objects."""

import time
from collections.abc import Callable
from typing import Any

from loguru import logger


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Count the time a function takes to execute."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger.info(f"Total execution time: {(end_time - start_time):.1f} seconds.")
        return result

    return wrapper
