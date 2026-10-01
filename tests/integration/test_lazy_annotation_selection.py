"""Lazy annotations must not make declaration mutations coverage-selectable."""

from __future__ import annotations

import json
import sys

import pytest


@pytest.mark.medium
@pytest.mark.skipif(sys.version_info < (3, 14), reason='Python 3.14 evaluates annotations lazily')
class DescribeLazyAnnotationSelection:
    def it_runs_all_tests_for_a_declaration_default_but_focuses_body_mutations(
        self,
        pytester_with_markers: pytest.Pytester,
    ) -> None:
        pytester_with_markers.makepyfile(
            target_module="""
from __future__ import annotations

from typing import Annotated

def is_enabled(version: Annotated[bool, 'version'] = False) -> bool:
    return version

def is_adult(age: int) -> bool:
    return age >= 18
"""
        )
        pytester_with_markers.makepyfile(
            test_target="""
from typing import get_type_hints

from target_module import is_adult, is_enabled

def test_a_explicit_version_resolves_lazy_signature():
    assert get_type_hints(is_enabled)['version'] == bool
    assert is_enabled(True) is True

def test_b_default_version_is_false():
    assert is_enabled() is False

def test_c_comparison_body_boundary():
    assert is_adult(18) is True
    assert is_adult(17) is False
"""
        )

        result = pytester_with_markers.runpytest_subprocess(
            '--gremlins',
            '--gremlin-targets=target_module.py',
            '--gremlin-operators=boolean,comparison',
            '--gremlin-no-lightweight-runner',
            '--gremlin-report=json',
            '-v',
        )

        assert result.ret == pytest.ExitCode.OK, result.stdout.str() + result.stderr.str()
        assert '3 passed' in result.stdout.str()
        report = json.loads((pytester_with_markers.path / 'coverage/gremlins/gremlins.json').read_text())
        default_mutant = next(row for row in report['results'] if row['description'] == 'False to True')
        assert default_mutant['status'] == 'zapped'
        assert default_mutant['selected_tests'] == [
            'test_target.py::test_a_explicit_version_resolves_lazy_signature',
            'test_target.py::test_b_default_version_is_false',
            'test_target.py::test_c_comparison_body_boundary',
        ]
        body_mutant = next(row for row in report['results'] if row['description'] == '>= to >')
        assert body_mutant['status'] == 'zapped'
        assert body_mutant['selected_tests'] == ['test_target.py::test_c_comparison_body_boundary']
