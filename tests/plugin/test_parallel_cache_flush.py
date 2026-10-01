"""Completed parallel mutant results are durable before the run ends."""

from __future__ import annotations

import ast
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from pytest_gremlins import plugin
from pytest_gremlins.cache.incremental import IncrementalCache
from pytest_gremlins.instrumentation.gremlin import Gremlin
from pytest_gremlins.parallel.pool import WorkerResult
from pytest_gremlins.plugin import GremlinSession, _run_parallel_mutation_testing
from pytest_gremlins.reporting.results import GremlinResultStatus


@pytest.mark.small
class DescribeParallelResultCacheFlush:
    def it_keeps_success_and_worker_error_results_before_pool_closes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        gremlin = Gremlin(
            gremlin_id='g001',
            file_path='src/module.py',
            line_number=1,
            original_node=ast.parse('x == 1', mode='eval').body,
            mutated_node=ast.parse('x != 1', mode='eval').body,
            operator_name='comparison',
            description='== to !=',
        )
        failed_gremlin = Gremlin(
            gremlin_id='g002',
            file_path='src/module.py',
            line_number=2,
            original_node=ast.parse('x == 2', mode='eval').body,
            mutated_node=ast.parse('x != 2', mode='eval').body,
            operator_name='comparison',
            description='== to !=',
        )
        cache = MagicMock(spec=IncrementalCache)
        state = GremlinSession(
            gremlins=[gremlin, failed_gremlin],
            cache_enabled=True,
            cache=cache,
            source_hashes={'src/module.py': 'source-hash'},
        )
        completed: Future[WorkerResult] = Future()
        completed.set_result(WorkerResult('g001', GremlinResultStatus.ZAPPED))
        failed: Future[WorkerResult] = Future()
        failed.set_exception(RuntimeError('worker failed'))
        futures = iter((completed, failed))

        class Pool:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                assert cache.flush.call_count == 2

            def submit(self, **_kwargs: object) -> Future[WorkerResult]:
                return next(futures)

        monkeypatch.setattr(plugin, '_get_rootdir', lambda _config: tmp_path)
        monkeypatch.setattr(plugin, '_build_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_build_filtered_test_command', lambda *_args: ['pytest'])
        monkeypatch.setattr(plugin, '_select_tests_for_gremlin_prioritized', lambda *_args: [])
        monkeypatch.setattr(plugin, '_check_cache_for_gremlin', lambda *_args: None)
        monkeypatch.setattr(plugin, 'WorkerPool', lambda **_kwargs: Pool())

        results = _run_parallel_mutation_testing(MagicMock(spec=pytest.Session), state)

        assert len(results) == 2
        assert {result.status for result in results} == {GremlinResultStatus.ZAPPED, GremlinResultStatus.ERROR}
        error_result = next(result for result in results if result.status == GremlinResultStatus.ERROR)
        assert error_result.error_output == 'worker failed'
        assert cache.cache_result_deferred.call_count == 2
        assert cache.flush.call_count == 2
