import functools
import time


def timelog(message):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            print(f"{message} started...")
            result = func(*args, **kwargs)
            print(f"{message} finished in {(time.perf_counter() - start):.3f} seconds")
            return result
        return wrapped
    return decorator
