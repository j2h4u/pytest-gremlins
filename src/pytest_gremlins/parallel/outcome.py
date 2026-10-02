"""Run mutation tests and classify only completed pytest receipts as outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from pytest_gremlins.parallel.process_runner import run_test_process
from pytest_gremlins.reporting.results import GremlinResultStatus

_OUTPUT_LIMIT = 2000


@dataclass(frozen=True)
class GremlinExecutionOutcome:
    """Native status and bounded diagnostics for one mutant test execution."""

    status: GremlinResultStatus
    error_output: str = ''


def run_gremlin_tests(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> GremlinExecutionOutcome:
    """Run tests, using an owned JUnit receipt for pytest outcome classification."""
    if not _is_pytest_command(command):
        try:
            return _classify_without_junit(
                run_test_process(command, cwd=cwd, env=env, timeout=timeout)
            )
        except subprocess.TimeoutExpired as exc:
            return GremlinExecutionOutcome(
                GremlinResultStatus.TIMEOUT,
                _format_output(exc.stdout, exc.stderr),
            )

    report_fd, report_name = tempfile.mkstemp(prefix='pytest-gremlins-', suffix='.xml')
    os.close(report_fd)
    report_path = Path(report_name)
    report_path.unlink()
    try:
        result = run_test_process(
            [*command, f'--junitxml={report_path}'],
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
        return _classify_junit_result(result, report_path)
    except subprocess.TimeoutExpired as exc:
        return GremlinExecutionOutcome(
            GremlinResultStatus.TIMEOUT,
            _format_output(exc.stdout, exc.stderr),
        )
    finally:
        report_path.unlink(missing_ok=True)


def _is_pytest_command(command: list[str]) -> bool:
    return 'pytest' in command or any(arg.endswith('gremlin_bootstrap.py') for arg in command)


def _classify_without_junit(
    result: subprocess.CompletedProcess[bytes],
) -> GremlinExecutionOutcome:
    if result.returncode == 0:
        return GremlinExecutionOutcome(GremlinResultStatus.SURVIVED)
    if result.returncode == 1:
        return GremlinExecutionOutcome(GremlinResultStatus.ZAPPED)
    return GremlinExecutionOutcome(
        GremlinResultStatus.ERROR,
        _format_output(result.stdout, result.stderr),
    )


def _classify_junit_result(
    result: subprocess.CompletedProcess[bytes], report_path: Path,
) -> GremlinExecutionOutcome:
    output = _format_output(result.stdout, result.stderr)
    if not report_path.is_file():
        return GremlinExecutionOutcome(
            GremlinResultStatus.ERROR,
            _with_reason(output, 'pytest did not write JUnit report'),
        )
    try:
        cases = ET.parse(report_path).getroot().iter('testcase')  # noqa: S314  # pytest created this owned report
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        for case in cases:
            counts['tests'] += 1
            for child in case:
                count_name = {'failure': 'failures', 'error': 'errors', 'skipped': 'skipped'}.get(child.tag)
                if count_name is not None:
                    counts[count_name] += 1
    except (ET.ParseError, OSError, ValueError) as exc:
        return GremlinExecutionOutcome(
            GremlinResultStatus.ERROR,
            _with_reason(output, f'invalid JUnit report: {exc}'),
        )

    if counts['tests'] == 0 or counts['tests'] == counts['skipped'] or counts['errors']:
        return GremlinExecutionOutcome(
            GremlinResultStatus.ERROR,
            _with_reason(output, 'JUnit report has no completed test outcomes'),
        )
    if result.returncode == 1 and counts['failures']:
        return GremlinExecutionOutcome(GremlinResultStatus.ZAPPED, output)
    if result.returncode == 0 and not counts['failures']:
        return GremlinExecutionOutcome(GremlinResultStatus.SURVIVED)
    return GremlinExecutionOutcome(
        GremlinResultStatus.ERROR,
        _with_reason(output, 'pytest exit status disagrees with JUnit report'),
    )


def _format_output(stdout: bytes | str | None, stderr: bytes | str | None) -> str:
    def decode(value: bytes | str | None) -> str:
        if value is None:
            return ''
        return value.decode(errors='replace') if isinstance(value, bytes) else value

    parts = [f'stdout:\n{decode(stdout)[:_OUTPUT_LIMIT]}', f'stderr:\n{decode(stderr)[:_OUTPUT_LIMIT]}']
    return '\n'.join(part for part in parts if part.split('\n', 1)[1])


def _with_reason(output: str, reason: str) -> str:
    return f'{reason}\n{output}' if output else reason
