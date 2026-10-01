"""The mutation bootstrap must retain normal package import metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from pytest_gremlins.plugin import _get_bootstrap_script


@pytest.mark.medium
class DescribeBootstrapPackageMetadata:
    def it_preserves_metadata_resources_and_relative_imports(
        self,
        pytester_with_markers: pytest.Pytester,
    ) -> None:
        package = pytester_with_markers.path / 'sample_package'
        package.mkdir()
        (package / '__init__.py').write_text(
            """
from importlib.resources import files
from pathlib import Path
from venv import EnvBuilder
from .helper import VALUE

def classify(value):
    if value >= 10:
        return 'large'
    return 'small'

def create_fresh_environment(destination):
    EnvBuilder(with_pip=False, clear=True).create(destination)

def metadata_is_intact():
    return (
        Path(__file__).is_file()
        and __spec__.origin == __file__
        and __spec__.has_location
        and __package__ == 'sample_package'
        and bool(__path__)
        and files(__package__).joinpath('marker.txt').read_text() == 'package resource'
        and VALUE == 'relative import'
    )
"""
        )
        (package / 'helper.py').write_text("VALUE = 'relative import'\n")
        (package / 'marker.txt').write_text('package resource')
        pytester_with_markers.makepyfile(
            test_sample_package="""
from sample_package import classify, create_fresh_environment, metadata_is_intact

def test_import_metadata_and_resource_access():
    assert metadata_is_intact()

def test_comparison_boundary():
    assert classify(10) == 'large'
    assert classify(9) == 'small'

def test_fresh_environment(tmp_path):
    destination = tmp_path / 'environment'
    create_fresh_environment(str(destination))
    assert (destination / 'bin' / 'python').exists()
"""
        )

        sources_file = pytester_with_markers.path / 'sources.json'
        sources_file.write_text(json.dumps({'sample_package': (package / '__init__.py').read_text()}))
        bootstrap_file = pytester_with_markers.path / 'gremlin_bootstrap.py'
        bootstrap_file.write_text(_get_bootstrap_script())
        env = os.environ.copy()
        env['PYTEST_GREMLINS_SOURCES_FILE'] = str(sources_file)
        env['ACTIVE_GREMLIN'] = 'no-match'
        baseline = subprocess.run(
            [sys.executable, str(bootstrap_file), '-p', 'no:gremlins', 'test_sample_package.py', '-q'],
            cwd=pytester_with_markers.path,
            env=env,
            capture_output=True,
            check=False,
            text=True,
        )
        assert baseline.returncode == 0, baseline.stdout + baseline.stderr
        assert '3 passed' in baseline.stdout

        result = pytester_with_markers.runpytest_subprocess(
            '--gremlins',
            '--gremlin-targets=sample_package/__init__.py',
            '--gremlin-operators=comparison,boolean',
            '--gremlin-no-lightweight-runner',
            '--gremlin-report=json',
            '-v',
        )

        output = result.stdout.str()
        assert result.ret == pytest.ExitCode.OK, result.stdout.str() + result.stderr.str()
        assert '3 passed' in output
        report = json.loads((pytester_with_markers.path / 'coverage/gremlins/gremlins.json').read_text())
        clear_mutant = next(row for row in report['results'] if row['description'] == 'True to False')
        assert clear_mutant['status'] == 'survived'
        comparison_mutant = next(row for row in report['results'] if row['description'] == '>= to >')
        assert comparison_mutant['status'] == 'zapped'
