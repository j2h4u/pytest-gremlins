"""Coverage selection preserves punctuation in parameterized node IDs."""

from __future__ import annotations

import json

import pytest


@pytest.mark.medium
class DescribeParameterizedNodeIdSelection:
    def it_zaps_a_mutant_using_the_exact_complex_parameter_node_id(
        self,
        pytester_with_markers: pytest.Pytester,
    ) -> None:
        pytester_with_markers.makepyfile(
            target_module="""
def can_continue(status):
    return status == 'pending' or status == 'ready'
"""
        )
        pytester_with_markers.makepyfile(
            test_target=r"""
import pytest

from target_module import can_continue

@pytest.mark.parametrize(
    'status',
    [pytest.param('ready', id=r'pending = ["address"]\n[ready]')],
)
def test_complex_parameter_is_the_killer(status):
    assert can_continue(status)

def test_unrelated():
    assert 1 + 1 == 2
"""
        )

        result = pytester_with_markers.runpytest_subprocess(
            '--gremlins',
            '--gremlin-targets=target_module.py',
            '--gremlin-operators=boolean',
            '--gremlin-no-lightweight-runner',
            '--gremlin-report=json',
            '-v',
        )

        assert result.ret == pytest.ExitCode.OK, result.stdout.str() + result.stderr.str()
        assert '2 passed' in result.stdout.str()
        report = json.loads((pytester_with_markers.path / 'coverage/gremlins/gremlins.json').read_text())
        mutant = next(row for row in report['results'] if row['description'] == 'or to and')
        assert mutant['status'] == 'zapped'
        assert mutant['selected_tests'] == [
            r'test_target.py::test_complex_parameter_is_the_killer[pending = ["address"]\\n[ready]]'
        ]
