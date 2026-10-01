"""Mutation selection safety follows Python's execution scopes."""

from __future__ import annotations

import sys

import pytest

from pytest_gremlins.instrumentation.transformer import collect_gremlins, transform_source
from pytest_gremlins.operators.boolean import BooleanOperator


@pytest.mark.small
@pytest.mark.parametrize(
    'source',
    [
        'FLAG = False',
        'class Config:\n    enabled = False',
        '@decorate(False)\ndef function():\n    return None',
        'def function(value=False):\n    return None',
        'def function(value: Literal[False]):\n    return None',
        'def function() -> Literal[False]:\n    return None',
        'class Config(base(False), metaclass=make(False)):\n    pass',
        'callback = lambda value=False: None',
        'def outer():\n    def inner(value=False):\n        return None',
        'def outer():\n    value: Literal[False] = None',
    ],
)
class DescribeExecutionTimeDeclarations:
    def it_marks_declaration_mutations_for_full_suite(self, source: str) -> None:
        gremlins, _tree = transform_source(source, 'example.py', [BooleanOperator()])

        assert gremlins
        assert all(gremlin.requires_full_suite for gremlin in gremlins)


@pytest.mark.small
@pytest.mark.parametrize(
    'source',
    [
        'def function():\n    value = False',
        'async def function():\n    value = False',
        'callback = lambda: False',
        'callback = lambda: (False if True else False)',
        'def outer():\n    def inner():\n        return False',
    ],
)
class DescribeCallableBodies:
    def it_keeps_callable_body_mutations_coverage_selectable(self, source: str) -> None:
        gremlins, _tree = transform_source(source, 'example.py', [BooleanOperator()])

        assert gremlins
        assert all(not gremlin.requires_full_suite for gremlin in gremlins)


@pytest.mark.small
class DescribeModernLazyDeclarations:
    @pytest.mark.skipif(sys.version_info < (3, 12), reason='PEP 695 type parameters require Python 3.12+')
    @pytest.mark.parametrize(
        'source',
        [
            'def function[T: False]():\n    return None',
            'class Config[T: False]:\n    pass',
            'type Alias[T: False] = int',
            'type Alias = Literal[False]',
        ],
    )
    def it_marks_type_parameter_and_lazy_alias_expressions(
        self,
        source: str,
    ) -> None:
        gremlins, _tree = transform_source(source, 'example.py', [BooleanOperator()])

        assert gremlins
        assert all(gremlin.requires_full_suite for gremlin in gremlins)


@pytest.mark.small
class DescribeGremlinScopePreservation:
    def it_keeps_full_suite_metadata_when_a_pragma_replaces_gremlins(self) -> None:
        source = 'FLAG = False  # gremlin: pardon[equivalent] intentionally static\n'

        gremlins, _tree = transform_source(source, 'example.py', [BooleanOperator()])

        assert gremlins
        assert all(gremlin.requires_full_suite for gremlin in gremlins)
        assert all(gremlin.pardoned for gremlin in gremlins)

    def it_marks_module_comparisons_in_collect_only_results(self) -> None:
        gremlins, _tree = collect_gremlins('if FLAG < 1:\n    pass', 'example.py')

        assert gremlins
        assert all(gremlin.requires_full_suite for gremlin in gremlins)

    def it_leaves_callable_body_comparisons_coverage_selectable_in_collect_only_results(self) -> None:
        gremlins, _tree = collect_gremlins('def is_small(value):\n    return value < 1', 'example.py')

        assert gremlins
        assert all(not gremlin.requires_full_suite for gremlin in gremlins)
