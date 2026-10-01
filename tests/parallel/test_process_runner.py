"""Adversarial process-tree timeout and parent-termination tests."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from pytest_gremlins.parallel.process_runner import run_test_process


def _wait_for_pids(*paths: Path) -> list[int]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all(path.exists() for path in paths):
            return [int(path.read_text()) for path in paths]
        time.sleep(0.02)
    pytest.fail(f'process pid files were not ready: {paths}')


def _assert_gone(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            state = Path(f'/proc/{pid}/stat').read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            return
        if state == 'Z':
            return
        time.sleep(0.02)
    pytest.fail(f'process {pid} is still running')


def _command_that_spawns_term_ignoring_grandchild(parent_pid_file: Path, grandchild_pid_file: Path) -> list[str]:
    grandchild = (
        'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        f"open({str(grandchild_pid_file)!r}, 'w').write(str(__import__('os').getpid())); "
        'time.sleep(30)'
    )
    parent = (
        'import os,subprocess,sys,time; '
        f"p=subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        f"open({str(parent_pid_file)!r}, 'w').write(str(os.getpid())); "
        "print('timeout stdout', flush=True); print('timeout stderr', file=sys.stderr, flush=True); "
        'time.sleep(30)'
    )
    return [sys.executable, '-c', parent]


@pytest.mark.medium
@pytest.mark.skipif(sys.platform != 'linux', reason='process-state checks use Linux /proc')
class DescribeProcessRunnerCleanup:
    """The runner cleans descendants on timeout and when its parent is terminated."""

    def it_kills_a_term_ignoring_grandchild_after_timeout(self, tmp_path: Path) -> None:
        parent_pid_file = tmp_path / 'parent-pid'
        grandchild_pid_file = tmp_path / 'grandchild-pid'
        with pytest.raises(subprocess.TimeoutExpired) as exc_info:
            run_test_process(
                _command_that_spawns_term_ignoring_grandchild(parent_pid_file, grandchild_pid_file),
                cwd=str(tmp_path),
                env=os.environ.copy(),
                timeout=2,
            )

        parent_pid, grandchild_pid = _wait_for_pids(parent_pid_file, grandchild_pid_file)
        assert exc_info.value.stdout == b'timeout stdout\n'
        assert exc_info.value.stderr == b'timeout stderr\n'
        _assert_gone(parent_pid)
        _assert_gone(grandchild_pid)

    @pytest.mark.parametrize('exit_code', [0, 7])
    def it_kills_a_term_ignoring_descendant_after_normal_or_error_exit(self, tmp_path: Path, exit_code: int) -> None:
        parent_pid_file = tmp_path / f'parent-{exit_code}'
        grandchild_pid_file = tmp_path / f'grandchild-{exit_code}'
        command = _command_that_exits_with_descendant(parent_pid_file, grandchild_pid_file, exit_code)
        result = run_test_process(command, cwd=str(tmp_path), env=os.environ.copy(), timeout=10)

        parent_pid, grandchild_pid = _wait_for_pids(parent_pid_file, grandchild_pid_file)
        assert result.returncode == exit_code
        assert result.stdout == b'child stdout\n'
        assert result.stderr == b'child stderr\n'
        _assert_gone(parent_pid)
        _assert_gone(grandchild_pid)

    def it_kills_active_groups_before_parent_sigterm_exits(self, tmp_path: Path) -> None:
        self._assert_parent_sigterm_cleans_group(tmp_path, during_spawn=False)

    def it_kills_a_group_when_parent_sigterm_arrives_during_spawn(self, tmp_path: Path) -> None:
        self._assert_parent_sigterm_cleans_group(tmp_path, during_spawn=True)

    def _assert_parent_sigterm_cleans_group(self, tmp_path: Path, *, during_spawn: bool) -> None:
        parent_pid_file = tmp_path / 'parent-pid'
        grandchild_pid_file = tmp_path / 'grandchild-pid'
        source_root = Path(__file__).resolve().parents[2]
        command = _command_that_spawns_term_ignoring_grandchild(parent_pid_file, grandchild_pid_file)
        runner_script = (
            'import os,sys,time\n'
            f'sys.path.insert(0, {str(source_root)!r})\n'
            'import pytest_gremlins.parallel.process_runner as runner\n'
        )
        if during_spawn:
            runner_script += (
                '_popen = runner.subprocess.Popen\n'
                'def _spawn_then_signal(*args, **kwargs):\n'
                '    process = _popen(*args, **kwargs)\n'
                '    deadline = time.monotonic() + 5\n'
                f'    while not os.path.exists({str(grandchild_pid_file)!r}) and time.monotonic() < deadline:\n'
                '        time.sleep(0.01)\n'
                '    os.kill(os.getpid(), 15)\n'
                '    return process\n'
                'runner.subprocess.Popen = _spawn_then_signal\n'
            )
        runner_script += f'runner.run_test_process({command!r}, cwd={str(tmp_path)!r}, env=None, timeout=30)\n'
        parent = subprocess.Popen(
            [sys.executable, '-c', runner_script],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            parent_pid, grandchild_pid = _wait_for_pids(parent_pid_file, grandchild_pid_file)
            parent.send_signal(signal.SIGTERM)
            parent.communicate(timeout=5)
            assert parent.returncode == -signal.SIGTERM
            _assert_gone(parent_pid)
            _assert_gone(grandchild_pid)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.communicate()


def _command_that_exits_with_descendant(parent_pid_file: Path, grandchild_pid_file: Path, exit_code: int) -> list[str]:
    grandchild = (
        'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        f"open({str(grandchild_pid_file)!r}, 'w').write(str(__import__('os').getpid())); "
        'time.sleep(30)'
    )
    parent = (
        'import os,subprocess,sys,time\n'
        f"p=subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
        'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
        f"open({str(parent_pid_file)!r}, 'w').write(str(os.getpid()))\n"
        f'deadline=time.monotonic()+5\n'
        f'while not os.path.exists({str(grandchild_pid_file)!r}) and time.monotonic() < deadline:\n'
        '    time.sleep(0.01)\n'
        "print('child stdout', flush=True)\n"
        "print('child stderr', file=sys.stderr, flush=True)\n"
        f'sys.exit({exit_code})\n'
    )
    return [sys.executable, '-c', parent]
