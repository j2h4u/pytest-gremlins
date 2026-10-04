"""Native pytest receipts prevent incomplete executions becoming verdicts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ast
from pathlib import Path
import subprocess
import sys

import pytest

from pytest_gremlins.parallel import outcome
from pytest_gremlins.parallel.outcome import run_gremlin_tests
from pytest_gremlins.parallel.pool import _run_gremlin_test
from pytest_gremlins.parallel.persistent_pool import _run_gremlin_batch
from pytest_gremlins.plugin import _test_gremlin
from pytest_gremlins.instrumentation.gremlin import Gremlin
from pytest_gremlins.reporting.results import GremlinResultStatus


def _pytest_case(tmp_path: Path, body: str, *, timeout: float = 5) -> outcome.GremlinExecutionOutcome:
    case = tmp_path / 'test_case.py'
    case.write_text(body, encoding='utf-8')
    return run_gremlin_tests(
        [sys.executable, '-m', 'pytest', '-q', str(case)],
        cwd=str(tmp_path),
        env={},
        timeout=timeout,
    )


@pytest.mark.medium
@pytest.mark.parametrize(
    ('body', 'expected'),
    [
        ('def test_case(): assert False', GremlinResultStatus.ZAPPED),
        ('def test_case(): assert True', GremlinResultStatus.SURVIVED),
        (
            'import pytest\ndef test_pass(): assert True\n@pytest.mark.skip\ndef test_skip(): pass',
            GremlinResultStatus.SURVIVED,
        ),
        (
            'import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError("setup")\ndef test_case(broken): pass',
            GremlinResultStatus.ERROR,
        ),
        (
            'import os, threading, time\ndef test_case():\n'
            '    print("inner timeout", flush=True)\n'
            '    threading.Timer(0.2, lambda: os._exit(1)).start()\n'
            '    time.sleep(10)',
            GremlinResultStatus.ERROR,
        ),
    ],
)
def it_completed_and_abrupt_pytest_outcomes(tmp_path: Path, body: str, expected: GremlinResultStatus) -> None:
    result = _pytest_case(tmp_path, body)
    assert result.status == expected
    if expected == GremlinResultStatus.ZAPPED:
        assert 'AssertionError' in result.error_output


@pytest.mark.medium
def it_outer_timeout_remains_timeout_with_output(tmp_path: Path) -> None:
    result = _pytest_case(
        tmp_path,
        'import time\ndef test_case():\n    print("outer timeout", flush=True)\n    time.sleep(10)',
        timeout=0.2,
    )
    assert result.status == GremlinResultStatus.TIMEOUT


@pytest.mark.medium
def it_thread_timeout_preserves_the_stalled_test_stack(tmp_path: Path) -> None:
    pytest.importorskip('pytest_timeout')
    case = tmp_path / 'test_stall.py'
    case.write_text(
        'import sys\nimport time\n'
        'def test_stalled_function():\n'
        '    print("stderr-prefix-" + "p" * 2400, file=sys.stderr, flush=True)\n'
        '    time.sleep(10)\n',
        encoding='utf-8',
    )
    result = run_gremlin_tests(
        [
            sys.executable,
            '-m',
            'pytest',
            '-q',
            '-s',
            '--timeout=0.2',
            '--timeout-method=thread',
            str(case),
        ],
        cwd=str(tmp_path),
        env={},
        timeout=15,
    )

    assert result.status == GremlinResultStatus.ERROR
    assert 'pytest did not write JUnit report' in result.error_output
    assert 'stderr-prefix-' in result.error_output
    assert 'test_stalled_function' in result.error_output
    assert 'time.sleep(10)' in result.error_output


@pytest.mark.medium
def it_owned_report_overrides_and_does_not_leave_user_report(tmp_path: Path) -> None:
    case = tmp_path / 'test_case.py'
    old_report = tmp_path / 'user.xml'
    case.write_text('def test_case(): assert True\n', encoding='utf-8')
    result = run_gremlin_tests(
        [sys.executable, '-m', 'pytest', '-q', f'--junitxml={old_report}', str(case)],
        cwd=str(tmp_path),
        env={},
        timeout=5,
    )

    assert result.status == GremlinResultStatus.SURVIVED
    assert not old_report.exists()


@pytest.mark.medium
def it_owned_junit_paths_are_unique_and_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    report_paths: list[Path] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        report_paths.append(report)
        report.write_text('<testsuite><testcase classname="x" name="y" /></testsuite>', encoding='utf-8')
        return subprocess.CompletedProcess(command, 0, b'', b'')

    monkeypatch.setattr(outcome, 'run_test_process', fake_run)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: run_gremlin_tests(['pytest'], cwd='.', env={}, timeout=1),
                range(4),
            )
        )

    assert all(result.status == GremlinResultStatus.SURVIVED for result in results)
    assert len(set(report_paths)) == 4
    assert all(not report.exists() for report in report_paths)


@pytest.mark.medium
@pytest.mark.parametrize(
    ('xml', 'returncode', 'expected'),
    [
        ('<testsuite><testcase name="x"><failure /></testcase></testsuite>', 1, GremlinResultStatus.ZAPPED),
        ('<broken', 1, GremlinResultStatus.ERROR),
        ('<testsuite><testcase name="x"><skipped /></testcase></testsuite>', 0, GremlinResultStatus.ERROR),
    ],
)
def it_only_complete_test_failure_is_zapped(
    tmp_path: Path,
    xml: str,
    returncode: int,
    expected: GremlinResultStatus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        report.write_text(xml, encoding='utf-8')
        return subprocess.CompletedProcess(command, returncode, b'', b'')

    monkeypatch.setattr(outcome, 'run_test_process', fake_run)
    result = run_gremlin_tests(['pytest'], cwd=str(tmp_path), env={}, timeout=1)
    assert result.status == expected


@pytest.mark.medium
@pytest.mark.parametrize(
    ('xml', 'returncode', 'expected'),
    [
        (None, 0, GremlinResultStatus.ERROR),
        ('<testsuite />', 0, GremlinResultStatus.ERROR),
        (
            '<testsuite><testcase name="x"><failure /><error /></testcase></testsuite>',
            1,
            GremlinResultStatus.ERROR,
        ),
    ],
)
def it_rejects_missing_empty_and_error_reports(
    tmp_path: Path,
    xml: str | None,
    returncode: int,
    expected: GremlinResultStatus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if xml is not None:
            report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
            report.write_text(xml, encoding='utf-8')
        return subprocess.CompletedProcess(command, returncode, b'test stdout', b'test stderr')

    monkeypatch.setattr(outcome, 'run_test_process', fake_run)
    result = run_gremlin_tests(['pytest'], cwd=str(tmp_path), env={}, timeout=1)
    assert result.status == expected
    assert 'test stdout' in result.error_output
    assert 'test stderr' in result.error_output


@pytest.mark.medium
def it_preserves_both_streams_on_outer_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(command, 1, output=b'timeout stdout', stderr=b'timeout stderr')

    monkeypatch.setattr(outcome, 'run_test_process', fake_run)
    result = run_gremlin_tests(['pytest'], cwd='.', env={}, timeout=1)
    assert result.status == GremlinResultStatus.TIMEOUT
    assert 'timeout stdout' in result.error_output
    assert 'timeout stderr' in result.error_output


@pytest.mark.small
def it_bounds_labeled_streams_and_keeps_prefix_and_tail() -> None:
    stdout = 'stdout-start-' + 's' * 5000 + 'stdout-end'
    stderr = 'stderr-start-' + 'e' * 5000 + 'stderr-end'
    formatted = outcome._format_output(stdout, stderr)
    stdout_part, stderr_tail = formatted.split('\nstderr:\n', maxsplit=1)
    stderr_part = 'stderr:\n' + stderr_tail

    assert len(stdout_part) <= 2000
    assert len(stderr_part) <= 2000
    assert 'stdout-start-' in stdout_part
    assert 'stdout-end' in stdout_part
    assert 'stderr-start-' in stderr_part
    assert 'stderr-end' in stderr_part
    assert '<output omitted>' in stdout_part
    assert '<output omitted>' in stderr_part
    assert outcome._format_output(b'bad-\xff', None) == 'stdout:\nbad-\ufffd'
    assert outcome._format_output(None, 'short stderr') == 'stderr:\nshort stderr'


@pytest.mark.medium
def it_adds_bounded_junit_error_context_without_changing_error_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_cases = ''.join(
        f'<testcase classname="class-{index}-{"c" * 100}-class-end-{index}" '
        f'name="canary-{index}-{"n" * 100}-name-end-{index}">'
        '<failure message="completed failure" />'
        f'<error type="SetupError-{index}-{"t" * 100}-type-end-{index}" '
        f'message="fixture setup failed {index}">'
        'TRACE-BEGIN-' + 't' * 3000 + f'TRACE-END-{index}' + '</error></testcase>'
        for index in range(5)
    )
    junit = f'<testsuite>{report_cases}</testsuite>'
    stdout = b'stdout-start-' + b's' * 5000 + b'stdout-end'
    stderr = b'stderr-start-' + b'e' * 5000 + b'stderr-end'

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        report.write_text(junit, encoding='utf-8')
        return subprocess.CompletedProcess(command, 1, stdout, stderr)

    monkeypatch.setattr(outcome, 'run_test_process', fake_run)
    result = run_gremlin_tests(['pytest'], cwd=str(tmp_path), env={}, timeout=1)

    assert result.status == GremlinResultStatus.ERROR
    assert len(result.error_output) <= 6000
    reason, _, remainder = result.error_output.partition('\n')
    assert len(reason) <= 256
    details, _, streams = remainder.partition('\nstdout:\n')
    assert len(details) <= 1500
    assert 'JUnit counts: tests=5 failures=5 errors=5 skipped=0' in details
    for index in range(3):
        assert f'class-end-{index}' in details
        assert f'name-end-{index}' in details
        assert f'type-end-{index}' in details
        assert f'fixture setup failed {index}' in details
        assert f'TRACE-END-{index}' in details
    assert 'name-end-3' not in details
    assert 'TRACE-BEGIN-' in details
    assert 'stdout-start-' in streams
    assert 'stdout-end' in streams
    assert 'stderr-start-' in streams
    assert 'stderr-end' in streams


@pytest.mark.small
def it_routes_serial_pool_and_batch_through_shared_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = outcome.GremlinExecutionOutcome(GremlinResultStatus.ZAPPED, 'stdout: receipt\nstderr:')
    monkeypatch.setattr('pytest_gremlins.plugin.run_gremlin_tests', lambda *_args, **_kwargs: expected)
    monkeypatch.setattr('pytest_gremlins.parallel.pool.run_gremlin_tests', lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        'pytest_gremlins.parallel.persistent_pool.run_gremlin_tests', lambda *_args, **_kwargs: expected
    )
    gremlin = Gremlin(
        gremlin_id='g001',
        file_path='source.py',
        line_number=1,
        original_node=ast.parse('x > 0').body[0].value,  # type: ignore[attr-defined]
        mutated_node=ast.parse('x >= 0').body[0].value,  # type: ignore[attr-defined]
        operator_name='comparison',
        description='> to >=',
    )

    serial = _test_gremlin(gremlin, ['python', '-c', 'pass'], tmp_path, instrumented_dir=None)
    pooled = _run_gremlin_test('g001', ['python', '-c', 'pass'], str(tmp_path), {}, 1)
    batched = _run_gremlin_batch(['g001'], ['python', '-c', 'pass'], str(tmp_path), {}, 1)[0]

    assert serial.status == pooled.status == batched.status == GremlinResultStatus.ZAPPED
    assert serial.error_output == pooled.error_output == batched.error_output == expected.error_output
    assert pooled.killing_test == batched.killing_test == 'unknown'
