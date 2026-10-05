"""Configured timeouts reach the subprocess mutation runner."""

from __future__ import annotations

import ast
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pytest_gremlins import plugin
from pytest_gremlins.instrumentation.gremlin import Gremlin
from pytest_gremlins.parallel.pool import WorkerResult
from pytest_gremlins.plugin import (
    GremlinSession,
    _effective_timeout,
    _run_mutation_testing,
    _run_parallel_mutation_testing,
)
from pytest_gremlins.reporting.results import GremlinResult, GremlinResultStatus


@pytest.mark.small
class DescribeSequentialTimeoutPropagation:
    def it_uses_full_suite_timeout_only_for_full_suite_mutants(self) -> None:
        full_suite = Gremlin(
            gremlin_id='g001',
            file_path='src/module.py',
            line_number=1,
            original_node=ast.parse('FLAG', mode='eval').body,
            mutated_node=ast.parse('not FLAG', mode='eval').body,
            operator_name='boolean',
            description='FLAG to not FLAG',
            requires_full_suite=True,
        )
        targeted = Gremlin(
            gremlin_id='g002',
            file_path='src/module.py',
            line_number=1,
            original_node=ast.parse('x == 1', mode='eval').body,
            mutated_node=ast.parse('x != 1', mode='eval').body,
            operator_name='comparison',
            description='== to !=',
        )
        state = GremlinSession(timeout=150, full_suite_timeout=600)

        assert _effective_timeout(full_suite, state) == 600
        assert _effective_timeout(targeted, state) == 150

        state.test_node_ids = {'test_a': 'tests/test_a.py::test_a', 'test_b': 'tests/test_b.py::test_b'}
        selector = MagicMock()
        state.prioritized_selector = selector
        selector.select_tests_prioritized.return_value = ['test_a']
        assert _effective_timeout(targeted, state) == 150

        for selection in (['test_a', 'test_b'], [], ['unresolved']):
            selector.select_tests_prioritized.return_value = selection
            assert _effective_timeout(targeted, state) == 600

        state.prioritized_selector = None
        assert _effective_timeout(targeted, state) == 600
        state.no_coverage_filter = True
        assert _effective_timeout(targeted, state) == 600
        state.full_suite_timeout = None
        assert _effective_timeout(full_suite, state) == 150

    def it_passes_full_suite_timeout_to_serial_subprocess_runner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        full_suite = Gremlin(
            gremlin_id='g001',
            file_path='src/module.py',
            line_number=1,
            original_node=ast.parse('FLAG', mode='eval').body,
            mutated_node=ast.parse('not FLAG', mode='eval').body,
            operator_name='boolean',
            description='FLAG to not FLAG',
            requires_full_suite=True,
        )
        uncovered = Gremlin(
            gremlin_id='g002',
            file_path='src/module.py',
            line_number=1,
            original_node=ast.parse('x == 1', mode='eval').body,
            mutated_node=ast.parse('x != 1', mode='eval').body,
            operator_name='comparison',
            description='== to !=',
        )
        selector = MagicMock()
        state = GremlinSession(
            gremlins=[full_suite, uncovered],
            timeout=150,
            full_suite_timeout=600,
            test_node_ids={'test_a': 'tests/test_a.py::test_a'},
            prioritized_selector=selector,
        )
        selector.select_tests_prioritized.return_value = []
        session = MagicMock(spec=pytest.Session)
        session.config.option.gremlin_executor = 'subprocess'
        received: list[float] = []

        def test_gremlin(gremlin: Gremlin, *_args: object, timeout: float, **_kwargs: object) -> GremlinResult:
            received.append(timeout)
            return GremlinResult(gremlin=gremlin, status=GremlinResultStatus.ZAPPED)

        monkeypatch.setattr(plugin, '_get_rootdir', lambda _config: tmp_path)
        monkeypatch.setattr(plugin, '_build_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_build_filtered_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_check_cache_for_gremlin', lambda *_args: None)
        monkeypatch.setattr(plugin, '_test_gremlin', test_gremlin)

        results = _run_mutation_testing(session, state)

        assert len(results) == 2
        assert received == [600, 600]

    def it_passes_full_suite_timeout_only_to_full_suite_parallel_jobs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        full_suite = Gremlin(
            gremlin_id='g001',
            file_path='src/module.py',
            line_number=1,
            original_node=ast.parse('FLAG', mode='eval').body,
            mutated_node=ast.parse('not FLAG', mode='eval').body,
            operator_name='boolean',
            description='FLAG to not FLAG',
            requires_full_suite=True,
        )
        targeted = Gremlin(
            gremlin_id='g002',
            file_path='src/module.py',
            line_number=2,
            original_node=ast.parse('x == 1', mode='eval').body,
            mutated_node=ast.parse('x != 1', mode='eval').body,
            operator_name='comparison',
            description='== to !=',
        )
        uncovered = Gremlin(
            gremlin_id='g003',
            file_path='src/module.py',
            line_number=3,
            original_node=ast.parse('x == 2', mode='eval').body,
            mutated_node=ast.parse('x != 2', mode='eval').body,
            operator_name='comparison',
            description='== to !=',
        )
        selector = MagicMock()
        state = GremlinSession(
            gremlins=[full_suite, targeted, uncovered],
            total_tests=2,
            timeout=150,
            full_suite_timeout=600,
            parallel_workers=2,
            test_node_ids={'test_a': 'tests/test_a.py::test_a', 'test_b': 'tests/test_b.py::test_b'},
            prioritized_selector=selector,
        )
        selector.select_tests_prioritized.side_effect = lambda gremlin: (
            ['test_a'] if gremlin.gremlin_id == 'g002' else []
        )
        session = MagicMock(spec=pytest.Session)
        received: list[float | None] = []

        class FakePool:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def __enter__(self) -> FakePool:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def submit(self, **kwargs: object) -> Future[WorkerResult]:
                gremlin_id = str(kwargs['gremlin_id'])
                timeout = kwargs.get('timeout')
                received.append(float(timeout) if isinstance(timeout, (int, float)) else None)
                future: Future[WorkerResult] = Future()
                future.set_result(WorkerResult(gremlin_id, GremlinResultStatus.SURVIVED))
                return future

        monkeypatch.setattr(plugin, 'WorkerPool', FakePool)
        monkeypatch.setattr(plugin, '_get_rootdir', lambda _config: tmp_path)
        monkeypatch.setattr(plugin, '_build_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_build_filtered_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_check_cache_for_gremlin', lambda *_args: None)

        results = _run_parallel_mutation_testing(session, state)

        assert len(results) == 3
        assert received == [600, 150, 600]

    def it_passes_configured_timeout_to_test_gremlin(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        gremlin = Gremlin(
            gremlin_id='g001',
            file_path='src/module.py',
            line_number=1,
            original_node=ast.parse('x == 1', mode='eval').body,
            mutated_node=ast.parse('x != 1', mode='eval').body,
            operator_name='comparison',
            description='== to !=',
        )
        state = GremlinSession(gremlins=[gremlin], timeout=88.5)
        session = MagicMock(spec=pytest.Session)
        session.config.option.gremlin_executor = 'subprocess'
        received: list[float] = []

        def test_gremlin(*_args: object, timeout: float, **_kwargs: object) -> GremlinResult:
            received.append(timeout)
            return GremlinResult(gremlin=gremlin, status=GremlinResultStatus.ZAPPED)

        monkeypatch.setattr(plugin, '_get_rootdir', lambda _config: tmp_path)
        monkeypatch.setattr(plugin, '_build_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_build_filtered_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_select_tests_for_gremlin_prioritized', lambda *_args: [])
        monkeypatch.setattr(plugin, '_check_cache_for_gremlin', lambda *_args: None)
        monkeypatch.setattr(plugin, '_test_gremlin', test_gremlin)

        results = _run_mutation_testing(session, state)

        assert len(results) == 1
        assert received == [88.5]
