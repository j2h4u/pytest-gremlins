"""Tests for incremental reuse of canonical native coverage data."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from unittest.mock import patch

from coverage import CoverageData
import pytest

from pytest_gremlins.cache.incremental import IncrementalCache
from pytest_gremlins.plugin import GremlinSession, _collect_coverage


def _make_session(root: Path) -> tuple[GremlinSession, Path, Path, IncrementalCache]:
    source_file = root / 'src' / 'module.py'
    test_file = root / 'tests' / 'test_module.py'
    source_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source_file.write_text('first = 1\nsecond = 2\n', encoding='utf-8')
    test_file.write_text('def test_first(): pass\n', encoding='utf-8')

    cache = IncrementalCache(root / '.gremlins_cache')
    session = GremlinSession(enabled=True, cache_enabled=True, cache=cache)
    session.source_files = {str(source_file): source_file.read_text(encoding='utf-8')}
    session.gremlins = [type('GremlinStub', (), {'file_path': str(source_file)})()]
    session.test_node_ids = {'tests/test_module.py::test_first': 'tests/test_module.py::test_first'}
    return session, source_file, test_file, cache


def _write_coverage_database(root: Path, source_file: Path, line: int, node_id: str) -> None:
    data = CoverageData(basename=root / '.coverage')
    data.set_context(f'{node_id}|run')
    data.add_lines({str(source_file): {line}})
    data.write()
    data.close()


def _run_with_lines(root: Path, source_file: Path, line: int):
    def run_test_process(_cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _write_coverage_database(root, source_file, line, 'tests/test_module.py::test_first')
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=b'', stderr=b'')

    return run_test_process


def _selected_lines(session: GremlinSession, source_file: Path) -> set[int]:
    assert session.coverage_collector is not None
    return {
        line
        for file_path, line in session.coverage_collector.coverage_map.locations()
        if file_path == str(source_file)
    }


@pytest.mark.medium
def it_identical_session_resumes_saved_coverage_without_collecting_again(tmp_path: Path) -> None:
    session, source_file, _test_file, cache = _make_session(tmp_path)
    try:
        with patch('pytest_gremlins.plugin.run_test_process', side_effect=_run_with_lines(tmp_path, source_file, 1)):
            _collect_coverage(session, tmp_path)
        assert _selected_lines(session, source_file) == {1}

        with patch(
            'pytest_gremlins.plugin.run_test_process',
            side_effect=_run_with_lines(tmp_path, source_file, 2),
        ) as run:
            _collect_coverage(session, tmp_path)

        run.assert_not_called()
        assert _selected_lines(session, source_file) == {1}
    finally:
        cache.close()


@pytest.mark.medium
@pytest.mark.parametrize('change', ['source', 'test', 'node_ids', 'settings'])
def it_coverage_inputs_invalidate_snapshot(tmp_path: Path, change: str) -> None:
    session, source_file, test_file, cache = _make_session(tmp_path)
    try:
        with patch('pytest_gremlins.plugin.run_test_process', side_effect=_run_with_lines(tmp_path, source_file, 1)):
            _collect_coverage(session, tmp_path)

        if change == 'source':
            source_file.write_text('first = 3\nsecond = 2\n', encoding='utf-8')
            session.source_files[str(source_file)] = source_file.read_text(encoding='utf-8')
        elif change == 'test':
            test_file.write_text('def test_first(): assert True\n', encoding='utf-8')
        elif change == 'node_ids':
            session.test_node_ids['tests/test_module.py::test_added'] = 'tests/test_module.py::test_added'
        else:
            session.preserved_addopts = '--import-mode=importlib'

        with patch(
            'pytest_gremlins.plugin.run_test_process',
            side_effect=_run_with_lines(tmp_path, source_file, 2),
        ) as run:
            _collect_coverage(session, tmp_path)

        run.assert_called_once()
        assert _selected_lines(session, source_file) == {2}
    finally:
        cache.close()


@pytest.mark.medium
def it_corrupt_snapshot_recollects_and_clear_removes_snapshot(tmp_path: Path) -> None:
    session, source_file, _test_file, cache = _make_session(tmp_path)
    snapshot = tmp_path / '.gremlins_cache' / 'coverage.sqlite'
    manifest = tmp_path / '.gremlins_cache' / 'coverage.json'
    try:
        with patch('pytest_gremlins.plugin.run_test_process', side_effect=_run_with_lines(tmp_path, source_file, 1)):
            _collect_coverage(session, tmp_path)
        snapshot.write_bytes(b'corrupt database')

        with patch(
            'pytest_gremlins.plugin.run_test_process',
            side_effect=_run_with_lines(tmp_path, source_file, 2),
        ) as run:
            _collect_coverage(session, tmp_path)

        run.assert_called_once()
        assert _selected_lines(session, source_file) == {2}

        cache.clear()
        assert not snapshot.exists()
        assert not manifest.exists()
    finally:
        cache.close()


@pytest.mark.medium
def it_snapshot_from_identical_checkout_at_another_root_is_not_reused(tmp_path: Path) -> None:
    first_root = tmp_path / 'first'
    second_root = tmp_path / 'second'
    first, first_source, _first_test, first_cache = _make_session(first_root)
    second, second_source, _second_test, second_cache = _make_session(second_root)
    try:
        with patch('pytest_gremlins.plugin.run_test_process', side_effect=_run_with_lines(first_root, first_source, 1)):
            _collect_coverage(first, first_root)

        for filename in ('coverage.sqlite', 'coverage.json'):
            shutil.copy2(first_root / '.gremlins_cache' / filename, second_root / '.gremlins_cache' / filename)

        with patch(
            'pytest_gremlins.plugin.run_test_process',
            side_effect=_run_with_lines(second_root, second_source, 2),
        ) as run:
            _collect_coverage(second, second_root)

        run.assert_called_once()
        assert _selected_lines(second, second_source) == {2}
    finally:
        first_cache.close()
        second_cache.close()


@pytest.mark.medium
def it_failed_or_empty_collection_does_not_publish_a_snapshot(tmp_path: Path) -> None:
    session, source_file, _test_file, cache = _make_session(tmp_path)
    snapshot = tmp_path / '.gremlins_cache' / 'coverage.sqlite'
    manifest = tmp_path / '.gremlins_cache' / 'coverage.json'
    try:
        with patch('pytest_gremlins.plugin.run_test_process', side_effect=_run_with_lines(tmp_path, source_file, 1)):
            _collect_coverage(session, tmp_path)
        session.preserved_addopts = '--import-mode=importlib'

        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout=b'failure', stderr=b'')
        with (
            patch('pytest_gremlins.plugin.run_test_process', return_value=failed),
            pytest.raises(RuntimeError, match='status 1'),
        ):
            _collect_coverage(session, tmp_path)
        assert snapshot.exists()
        assert manifest.exists()

        empty = subprocess.CompletedProcess(args=[], returncode=0, stdout=b'', stderr=b'')
        with (
            patch('pytest_gremlins.plugin.run_test_process', return_value=empty),
            pytest.warns(UserWarning, match='Coverage collection returned no data'),
        ):
            _collect_coverage(session, tmp_path)
        with patch(
            'pytest_gremlins.plugin.run_test_process',
            side_effect=_run_with_lines(tmp_path, source_file, 2),
        ) as run:
            _collect_coverage(session, tmp_path)
        run.assert_called_once()
        assert _selected_lines(session, source_file) == {2}
    finally:
        cache.close()
