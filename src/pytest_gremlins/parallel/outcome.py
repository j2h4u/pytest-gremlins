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
_JUNIT_DETAILS_LIMIT = 1500
_MAX_JUNIT_ERRORS = 3
_REASON_LIMIT = 256
_ERROR_OUTPUT_LIMIT = 6000
_OMISSION_MARKER = '\n...<output omitted>...\n'


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
            return _classify_without_junit(run_test_process(command, cwd=cwd, env=env, timeout=timeout))
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
    result: subprocess.CompletedProcess[bytes],
    report_path: Path,
) -> GremlinExecutionOutcome:
    output = _format_output(result.stdout, result.stderr)
    if not report_path.is_file():
        return GremlinExecutionOutcome(
            GremlinResultStatus.ERROR,
            _with_reason(output, 'pytest did not write JUnit report'),
        )
    try:
        root = ET.parse(report_path).getroot()  # noqa: S314  # pytest created this owned report
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        error_cases: list[tuple[str, str, str, str, str]] = []
        for case in root.iter('testcase'):
            counts['tests'] += 1
            for child in case:
                count_name = {'failure': 'failures', 'error': 'errors', 'skipped': 'skipped'}.get(child.tag)
                if count_name is not None:
                    counts[count_name] += 1
                if child.tag == 'error' and len(error_cases) < _MAX_JUNIT_ERRORS:
                    error_cases.append(
                        (
                            case.get('classname', ''),
                            case.get('name', ''),
                            child.get('type', ''),
                            child.get('message', ''),
                            ''.join(child.itertext()),
                        )
                    )
    except (ET.ParseError, OSError, ValueError) as exc:
        return GremlinExecutionOutcome(
            GremlinResultStatus.ERROR,
            _with_reason(output, f'invalid JUnit report: {exc}'),
        )

    details = _format_junit_errors(counts, error_cases)
    if counts['tests'] == 0 or counts['tests'] == counts['skipped'] or counts['errors']:
        return GremlinExecutionOutcome(
            GremlinResultStatus.ERROR,
            _with_reason(output, 'JUnit report has no completed test outcomes', details=details),
        )
    if result.returncode == 1 and counts['failures']:
        return GremlinExecutionOutcome(GremlinResultStatus.ZAPPED, output)
    if result.returncode == 0 and not counts['failures']:
        return GremlinExecutionOutcome(GremlinResultStatus.SURVIVED)
    return GremlinExecutionOutcome(
        GremlinResultStatus.ERROR,
        _with_reason(output, 'pytest exit status disagrees with JUnit report', details=details),
    )


def _format_output(stdout: bytes | str | None, stderr: bytes | str | None) -> str:
    def decode(value: bytes | str | None) -> str:
        if value is None:
            return ''
        return value.decode(errors='replace') if isinstance(value, bytes) else value

    parts = [_format_stream('stdout', decode(stdout)), _format_stream('stderr', decode(stderr))]
    return '\n'.join(part for part in parts if part)


def _format_stream(label: str, value: str) -> str:
    if not value:
        return ''
    prefix = f'{label}:\n'
    return prefix + _bounded_text(value, _OUTPUT_LIMIT - len(prefix))


def _format_junit_errors(counts: dict[str, int], cases: list[tuple[str, str, str, str, str]]) -> str:
    summary = (
        f'JUnit counts: tests={counts["tests"]} failures={counts["failures"]} '
        f'errors={counts["errors"]} skipped={counts["skipped"]}'
    )
    if not cases:
        return summary

    case_parts: list[tuple[str, str]] = []
    for index, (classname, name, error_type, message, trace) in enumerate(cases, start=1):
        details = '\n'.join(
            (
                f'error {index}:',
                f'  classname={_bounded_text(classname, 64)}',
                f'  name={_bounded_text(name, 64)}',
                f'  type={_bounded_text(error_type, 64)}',
                f'  message={_bounded_text(message, 64)}',
            )
        )
        trace_prefix = f'{details}\n  traceback:\n'
        case_parts.append((trace_prefix, trace))

    base_length = len(summary) + len(case_parts) + sum(len(prefix) for prefix, _ in case_parts)
    trace_limit = max(1, (_JUNIT_DETAILS_LIMIT - base_length) // len(case_parts))
    rendered = [summary]
    rendered.extend(prefix + _bounded_text(trace, trace_limit) for prefix, trace in case_parts)
    return _bounded_text('\n'.join(rendered), _JUNIT_DETAILS_LIMIT)


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    available = limit - len(_OMISSION_MARKER)
    prefix_length = available // 2
    suffix_length = available - prefix_length
    return value[:prefix_length] + _OMISSION_MARKER + value[-suffix_length:]


def _with_reason(output: str, reason: str, *, details: str = '') -> str:
    parts = [_bounded_text(reason, _REASON_LIMIT)]
    if details:
        parts.append(details)
    if output:
        parts.append(output)
    return _bounded_text('\n'.join(parts), _ERROR_OUTPUT_LIMIT)
