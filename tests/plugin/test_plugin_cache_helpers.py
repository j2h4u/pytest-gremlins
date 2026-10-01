"""Tests for plugin cache and subprocess helper functions.

These tests cover the cache-related and subprocess helper functions in plugin.py,
including test hash building, cache lookup/store, and gremlin subprocess env vars.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from pytest_gremlins.cache.incremental import IncrementalCache
from pytest_gremlins.plugin import (
    GREMLIN_SOURCES_ENV_VAR,
    GremlinSession,
    _build_test_hashes_for_gremlin,
    _cache_gremlin_result,
    _check_cache_for_gremlin,
    _cache_run_config,
    _test_gremlin,
)
from pytest_gremlins.reporting.results import (
    GremlinResult,
    GremlinResultStatus,
)


@pytest.mark.small
class DescribeBuildTestHashesForGremlin:
    """Tests for _build_test_hashes_for_gremlin function.

    Tests the dotted-name fallback path (lines 1467-1468) where a test name
    like 'SomeClass.test_method' is resolved to 'test_method' to find its node ID.
    """

    def it_resolves_dotted_name_via_simple_name_fallback(self) -> None:
        """'Module.test_method' falls back to 'test_method' when the full name is not in test_node_ids."""
        gs = GremlinSession(
            enabled=True,
            test_node_ids={'test_method': 'tests/test_m.py::test_method'},
            test_hashes={'tests/test_m.py': 'abc123'},
        )

        result = _build_test_hashes_for_gremlin(['Module.test_method'], gs)

        assert 'Module.test_method' in result
        assert result['Module.test_method'] == 'abc123'

    def it_returns_empty_when_node_id_not_found_even_after_fallback(self) -> None:
        """Test names with no matching node ID (even via simple name) produce no entry."""
        gs = GremlinSession(enabled=True, test_node_ids={}, test_hashes={})

        result = _build_test_hashes_for_gremlin(['unknown.test_func'], gs)

        assert result == {}


@pytest.mark.small
class DescribeCheckCacheForGremlin:
    """Tests for _check_cache_for_gremlin function (lines 1498-1519).

    Both the None-return paths and the hit path are tested so no single
    hardcoded return value satisfies all cases.
    """

    def it_returns_none_when_cache_disabled(self) -> None:
        """Returns None immediately when cache_enabled is False."""
        gs = GremlinSession(enabled=True, cache_enabled=False)
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok

        result = _check_cache_for_gremlin(gremlin, [], gs)

        assert result is None

    def it_returns_none_when_source_hash_missing(self) -> None:
        """Returns None when gremlin's file_path has no entry in source_hashes."""
        mock_cache = MagicMock(spec=IncrementalCache)
        gs = GremlinSession(enabled=True, cache_enabled=True, cache=mock_cache, source_hashes={})
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok
        gremlin.file_path = 'src/module.py'

        result = _check_cache_for_gremlin(gremlin, [], gs)

        assert result is None
        mock_cache.get_cached_result.assert_not_called()

    def it_returns_gremlin_result_on_cache_hit(self) -> None:
        """Returns a GremlinResult constructed from cached data when cache has a hit."""
        mock_cache = MagicMock(spec=IncrementalCache)
        mock_cache.get_cached_result.return_value = {'status': 'zapped', 'killing_test': 'test_foo'}
        gs = GremlinSession(
            enabled=True,
            cache_enabled=True,
            cache=mock_cache,
            source_hashes={'src/module.py': 'hash123'},
        )
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok
        gremlin.file_path = 'src/module.py'

        result = _check_cache_for_gremlin(gremlin, [], gs)

        assert result is not None
        assert result.status == GremlinResultStatus.ZAPPED

    def it_keys_the_lookup_on_the_runner_mode(self) -> None:
        mock_cache = MagicMock(spec=IncrementalCache)
        mock_cache.get_cached_result.return_value = None
        gs = GremlinSession(
            enabled=True,
            cache_enabled=True,
            cache=mock_cache,
            source_hashes={'src/module.py': 'hash123'},
            lightweight_runner=False,
        )
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok
        gremlin.file_path = 'src/module.py'

        _check_cache_for_gremlin(gremlin, [], gs)

        assert mock_cache.get_cached_result.call_args.kwargs['run_config'] == (
            'lightweight_runner=False;timeout=30;operators='
        )

    def it_separates_cache_entries_by_timeout_and_operators(self) -> None:
        first = GremlinSession(timeout=30, operators=[])
        second = GremlinSession(timeout=120, operators=[])
        operator = MagicMock()
        operator.name = 'comparison'
        third = GremlinSession(timeout=30, operators=[operator])

        assert len({_cache_run_config(first), _cache_run_config(second), _cache_run_config(third)}) == 3

    def it_misses_when_operator_order_changes(self) -> None:
        comparison = MagicMock()
        comparison.name = 'comparison'
        arithmetic = MagicMock()
        arithmetic.name = 'arithmetic'
        cache = MagicMock(spec=IncrementalCache)
        first = GremlinSession(
            enabled=True,
            cache_enabled=True,
            cache=cache,
            source_hashes={'src/module.py': 'hash123'},
            operators=[comparison, arithmetic],
        )
        reversed_order = GremlinSession(
            enabled=True,
            cache_enabled=True,
            cache=cache,
            source_hashes={'src/module.py': 'hash123'},
            operators=[arithmetic, comparison],
        )
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; bare-mock: ok
        gremlin.file_path = 'src/module.py'
        cache.get_cached_result.side_effect = lambda **kwargs: (
            {'status': 'zapped'} if kwargs['run_config'] == _cache_run_config(first) else None
        )

        assert _check_cache_for_gremlin(gremlin, [], first) is not None
        assert _check_cache_for_gremlin(gremlin, [], reversed_order) is None


@pytest.mark.small
class DescribeCacheGremlinResult:
    """Tests for _cache_gremlin_result function (lines 1536-1555).

    Verifies that cache.cache_result_deferred is called when a source hash exists,
    and skipped when it does not.
    """

    def it_calls_cache_deferred_when_source_hash_exists(self) -> None:
        """cache_result_deferred is called when gremlin's file has a source hash."""
        mock_cache = MagicMock(spec=IncrementalCache)
        gs = GremlinSession(
            enabled=True,
            cache_enabled=True,
            cache=mock_cache,
            source_hashes={'src/module.py': 'hash123'},
        )
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok
        gremlin.file_path = 'src/module.py'
        gremlin.gremlin_id = 'g001'
        result = GremlinResult(gremlin=gremlin, status=GremlinResultStatus.ZAPPED)

        _cache_gremlin_result(gremlin, [], result, gs)

        mock_cache.cache_result_deferred.assert_called_once()
        assert mock_cache.cache_result_deferred.call_args.kwargs['run_config'] == (
            'lightweight_runner=True;timeout=30;operators='
        )

    def it_skips_caching_when_source_hash_missing(self) -> None:
        """cache_result_deferred is NOT called when gremlin's file has no source hash."""
        mock_cache = MagicMock(spec=IncrementalCache)
        gs = GremlinSession(
            enabled=True,
            cache_enabled=True,
            cache=mock_cache,
            source_hashes={},
        )
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok
        gremlin.file_path = 'src/module.py'
        result = GremlinResult(gremlin=gremlin, status=GremlinResultStatus.ZAPPED)

        _cache_gremlin_result(gremlin, [], result, gs)

        mock_cache.cache_result_deferred.assert_not_called()


@pytest.mark.small
class DescribeGremlinSubprocessEnvVars:
    """Tests for _test_gremlin function (lines 1737-1739).

    Verifies that GREMLIN_SOURCES_ENV_VAR is injected into the subprocess env
    when instrumented_dir is not None, but not when it is None.
    """

    def it_sets_sources_env_var_when_instrumented_dir_provided(self, tmp_path: Path) -> None:
        """GREMLIN_SOURCES_ENV_VAR is set to '<instrumented_dir>/sources.json' in env."""
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok
        gremlin.gremlin_id = 'g001'
        captured_env: dict[str, str] = {}

        def capture_env(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            env = kwargs.get('env')
            if isinstance(env, dict):
                captured_env.update(env)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.run_test_process', side_effect=capture_env):
            _test_gremlin(gremlin, ['pytest'], tmp_path, instrumented_dir=tmp_path)

        assert GREMLIN_SOURCES_ENV_VAR in captured_env
        assert captured_env[GREMLIN_SOURCES_ENV_VAR] == str(tmp_path / 'sources.json')

    def it_omits_sources_env_var_when_instrumented_dir_is_none(self, tmp_path: Path) -> None:
        """GREMLIN_SOURCES_ENV_VAR is NOT set when instrumented_dir is None."""
        gremlin = MagicMock()  # Gremlin is a frozen dataclass; spec= misses instance fields; bare-mock: ok
        gremlin.gremlin_id = 'g001'
        captured_env: dict[str, str] = {}

        def capture_env(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            env = kwargs.get('env')
            if isinstance(env, dict):
                captured_env.update(env)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b'', stderr=b'')

        with patch('pytest_gremlins.plugin.run_test_process', side_effect=capture_env):
            _test_gremlin(gremlin, ['pytest'], tmp_path, instrumented_dir=None)

        assert GREMLIN_SOURCES_ENV_VAR not in captured_env
