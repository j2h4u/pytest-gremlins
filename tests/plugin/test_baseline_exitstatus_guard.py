"""Mutation testing must stop when pytest's baseline session is not green."""

from __future__ import annotations

from contextlib import closing
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from pytest_gremlins.plugin import GremlinSession, _set_session, pytest_sessionfinish


@pytest.mark.small
@pytest.mark.parametrize(
    'exitstatus',
    [
        pytest.ExitCode.TESTS_FAILED,
        pytest.ExitCode.INTERRUPTED,
        pytest.ExitCode.INTERNAL_ERROR,
        pytest.ExitCode.USAGE_ERROR,
        pytest.ExitCode.NO_TESTS_COLLECTED,
    ],
)
def it_skips_coverage_and_mutation_after_nonzero_baseline(exitstatus: pytest.ExitCode) -> None:
    session = MagicMock(spec=pytest.Session)
    session.config = MagicMock(spec=[])
    gremlin_session = GremlinSession(enabled=True, gremlins=[MagicMock()])
    _set_session(gremlin_session)

    with (
        patch('pytest_gremlins.plugin._collect_coverage') as collect,
        patch('pytest_gremlins.plugin._dispatch_mutation_run') as dispatch,
    ):
        pytest_sessionfinish(session, exitstatus)

    collect.assert_not_called()
    dispatch.assert_not_called()


@pytest.mark.medium
def it_keeps_the_original_baseline_failure_without_snapshot_or_results(pytester: pytest.Pytester) -> None:
    pytester.makepyprojecttoml(
        """
        [tool.pytest-gremlins]
        paths = ["."]
        cache = true
        workers = 1
        """
    )
    pytester.makepyfile(
        target_module='def add(left, right):\n    return left + right\n',
        test_baseline_failure=(
            'from target_module import add\n\n'
            'def test_baseline_failure():\n'
            "    assert add(1, 2) == 99, 'baseline failure sentinel'\n"
        ),
    )

    result = pytester.runpytest_subprocess('--gremlins', '--gremlin-targets=target_module.py', '-q')
    output = result.stdout.str() + result.stderr.str()
    assert result.ret == pytest.ExitCode.TESTS_FAILED, output
    assert 'baseline failure sentinel' in output
    assert '1 failed' in output
    assert 'Coverage subprocess' not in output

    cache_dir = pytester.path / '.gremlins_cache'
    assert not (cache_dir / 'coverage.sqlite').exists()
    assert not (cache_dir / 'coverage.json').exists()
    results_path = cache_dir / 'results.db'
    if results_path.exists():
        with closing(sqlite3.connect(results_path)) as connection:
            assert connection.execute('SELECT COUNT(*) FROM results').fetchone() == (0,)
