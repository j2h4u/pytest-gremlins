"""Tests for multi-format report dispatch: _write_json_report and format routing."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from pytest_gremlins.plugin import (
    GremlinSession,
    _generate_gremlins,
    _write_json_report,
)
from pytest_gremlins.instrumentation.transformer import get_default_registry
from pytest_gremlins.reporting.score import MutationScore


def _empty_score() -> MutationScore:
    return MutationScore.from_results([])


@pytest.mark.medium
class DescribeWriteJsonReport:
    """_write_json_report writes JSON to coverage/gremlins/gremlins.json."""

    def it_writes_to_coverage_gremlins_gremlins_json(self, tmp_path: Path) -> None:
        score = _empty_score()

        result_path = _write_json_report(score, rootdir=tmp_path)

        assert result_path == tmp_path / 'coverage' / 'gremlins' / 'gremlins.json'
        assert result_path.exists()

    def it_creates_coverage_gremlins_directory_automatically(self, tmp_path: Path) -> None:
        score = _empty_score()
        expected_dir = tmp_path / 'coverage' / 'gremlins'
        assert not expected_dir.exists()

        _write_json_report(score, rootdir=tmp_path)

        assert expected_dir.is_dir()

    def it_writes_valid_json(self, tmp_path: Path) -> None:
        score = _empty_score()

        result_path = _write_json_report(score, rootdir=tmp_path)

        data = json.loads(result_path.read_text())
        assert 'summary' in data
        assert 'results' in data

    def it_includes_native_scope_when_available(self, tmp_path: Path) -> None:
        result_path = _write_json_report(
            _empty_score(),
            rootdir=tmp_path,
            source_files={
                str(tmp_path / 'src' / 'a.py'): '',
                str(tmp_path / 'src' / 'b.py'): '',
                str(tmp_path / 'src' / '..' / 'src' / 'a.py'): '',
            },
            gremlin_ids=['g002', 'g001'],
            generation_errors=[str(tmp_path / 'src' / 'a.py')],
        )

        assert json.loads(result_path.read_text())['scope'] == {
            'source_files': ['src/a.py', 'src/b.py'],
            'gremlin_ids': ['g001', 'g002'],
            'generation_errors': ['src/a.py'],
        }


@pytest.mark.medium
class DescribeGremlinSessionReportFormats:
    """GremlinSession.report_formats defaults to ['console']."""

    def it_defaults_report_formats_to_console_list(self) -> None:
        session = GremlinSession()

        assert session.report_formats == ['console']

    def it_accepts_multiple_formats(self) -> None:
        session = GremlinSession(report_formats=['html', 'json'])

        assert session.report_formats == ['html', 'json']

    def it_retains_native_generation_failures(self, tmp_path: Path) -> None:
        session = GremlinSession()
        source_path = str(tmp_path / 'source.py')
        with patch('pytest_gremlins.plugin.transform_source', side_effect=RuntimeError('bad source')):
            _generate_gremlins(session, {source_path: 'bad'}, tmp_path)

        assert session.generation_errors == [source_path]

    @pytest.mark.medium
    def it_tracks_native_sources_and_generated_ids(self, tmp_path: Path) -> None:
        sources = {
            str(tmp_path / 'src' / 'with_mutation.py'): 'def check(value): return value > 0\n',
            str(tmp_path / 'src' / 'without_mutation.py'): 'VALUE = 1\n',
        }
        session = GremlinSession(operators=get_default_registry().get_all(), source_files=sources)

        _generate_gremlins(session, sources, tmp_path)
        try:
            assert set(session.source_files) == set(sources)
            assert session.gremlins
            assert session.generation_errors == []
            report = _write_json_report(
                _empty_score(),
                tmp_path,
                source_files=session.source_files,
                gremlin_ids=[gremlin.gremlin_id for gremlin in session.gremlins],
                generation_errors=session.generation_errors,
            )
            scope = json.loads(report.read_text())['scope']
            assert scope['source_files'] == ['src/with_mutation.py', 'src/without_mutation.py']
            assert scope['gremlin_ids'] == sorted(gremlin.gremlin_id for gremlin in session.gremlins)
            assert scope['generation_errors'] == []
        finally:
            if session.instrumented_dir is not None:
                shutil.rmtree(session.instrumented_dir)
