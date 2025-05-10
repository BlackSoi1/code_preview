from typing import Optional, Dict, List, Any
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile
import warnings

warnings.filterwarnings("ignore")


def unsafe_execute(program: str, timeout: float, result: List[str]) -> None:
    """Execute the program in a sandboxed environment."""
    with create_tempdir():
        warnings.filterwarnings("ignore")
        try:
            exec_globals = {}
            with swallow_io(), time_limit(timeout):
                exec(program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except AssertionError as e:
            result.append(f"Assertion failed: {e}")
        except Exception as e:
            result.append(f"failed: {e}")


def check_correctness(
    program: str, timeout: float, completion_id: Optional[int] = None
) -> Dict[str, Any]:
    """Check the correctness of a program by running it in a separate process."""
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(program, timeout, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return {
        "passed": result[0] == "passed",
        "result": result[0],
        "completion_id": completion_id,
    }


@contextlib.contextmanager
def time_limit(seconds: float):
    """Set a time limit for code execution."""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    """Redirect and swallow stdout, stderr, and stdin."""
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(
        stream
    ), redirect_stdin(stream):
        yield


@contextlib.contextmanager
def create_tempdir():
    """Create and change to a temporary directory."""
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError("Cannot read from WriteOnlyStringIO")

    def readline(self, *args, **kwargs):
        raise IOError("Cannot read from WriteOnlyStringIO")

    def readlines(self, *args, **kwargs):
        raise IOError("Cannot read from WriteOnlyStringIO")

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):
    """Context manager for temporarily redirecting stdin."""

    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    """Change directory and return to the original directory afterward."""
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)
