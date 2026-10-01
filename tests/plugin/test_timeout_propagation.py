"""Configured timeouts reach the subprocess mutation runner."""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from pytest_gremlins import plugin
from pytest_gremlins.instrumentation.gremlin import Gremlin
from pytest_gremlins.plugin import GremlinSession, _run_mutation_testing
from pytest_gremlins.reporting.results import GremlinResult, GremlinResultStatus


@pytest.mark.small
class DescribeSequentialTimeoutPropagation:
    def it_passes_configured_timeout_to_test_gremlin(
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
