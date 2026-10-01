"""Run test subprocesses with timeout cleanup for their process groups."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from types import FrameType

_SignalHandler = signal.Handlers | Callable[[int, FrameType | None], object]

_TERM_GRACE_SECONDS = 0.5


def run_test_process(
    args: list[str],
    *,
    cwd: str,
    env: dict[str, str] | None,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run a test command, terminating its POSIX process tree on timeout."""
    if os.name != 'posix':
        return subprocess.run(args, cwd=cwd, env=env, capture_output=True, timeout=timeout, check=False)
    if threading.current_thread() is not threading.main_thread():
        msg = 'POSIX test subprocesses must run on the process main thread'
        raise RuntimeError(msg)

    previous_handler: _SignalHandler = signal.getsignal(signal.SIGTERM)
    process: subprocess.Popen[bytes] | None = None
    termination_requested = False

    def handle_sigterm(signum: int, frame: FrameType | None) -> None:
        nonlocal termination_requested
        signal.signal(signum, signal.SIG_IGN)
        if process is None:
            termination_requested = True
            return
        _stop_process_group(process.pid)
        _restore_and_reraise(signum, frame, previous_handler)

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        try:
            process = subprocess.Popen(
                args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except BaseException:
            if termination_requested:
                _restore_and_reraise(signal.SIGTERM, None, previous_handler)
            raise
        if termination_requested:
            _stop_process_group(process.pid)
            process.communicate()
            _restore_and_reraise(signal.SIGTERM, None, previous_handler)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _stop_process_group(process.pid)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                args,
                timeout,
                output=stdout if stdout is not None else exc.output,
                stderr=stderr if stderr is not None else exc.stderr,
            ) from exc

        _cleanup_descendants(process)
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    finally:
        if process is not None:
            _reap_process(process)
        signal.signal(signal.SIGTERM, previous_handler)


def _restore_and_reraise(signum: int, frame: FrameType | None, previous_handler: _SignalHandler) -> None:
    signal.signal(signum, previous_handler)
    if callable(previous_handler):
        previous_handler(signum, frame)
    elif previous_handler == signal.SIG_DFL:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _stop_process_group(process_group: int) -> None:
    _signal_process_group(process_group, signal.SIGTERM)
    deadline = time.monotonic() + _TERM_GRACE_SECONDS
    while time.monotonic() < deadline and _process_group_exists(process_group):
        time.sleep(0.02)
    if _process_group_exists(process_group):
        _signal_process_group(process_group, signal.SIGKILL)


def _cleanup_descendants(process: subprocess.Popen[bytes]) -> None:
    if _process_group_exists(process.pid):
        _stop_process_group(process.pid)
        process.communicate()


def _reap_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None or _process_group_exists(process.pid):
        _stop_process_group(process.pid)
    process.communicate()


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_process_group(process_group: int, sig: signal.Signals) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process_group, sig)
