"""Tests for --gremlin-no-lightweight-runner: the option, the session field, and the merged value."""

from __future__ import annotations

from unittest.mock import MagicMock

from _pytest.config.argparsing import (
    OptionGroup,
    Parser,
)
import pytest

from pytest_gremlins.config import GremlinConfig
from pytest_gremlins.plugin import (
    GremlinSession,
    _extract_toml_fields,
    pytest_addoption,
)


@pytest.mark.small
class DescribeNoLightweightRunnerOption:
    def it_registers_the_option_as_a_flag_defaulting_off(self) -> None:
        parser = MagicMock(spec=Parser)
        group = MagicMock(spec=OptionGroup)
        parser.getgroup.return_value = group

        pytest_addoption(parser)

        added_options = {c.args[0]: c.kwargs for c in group.addoption.call_args_list if c.args}
        option = added_options['--gremlin-no-lightweight-runner']
        assert option['action'] == 'store_true'
        assert option['default'] is False
        assert option['dest'] == 'gremlin_no_lightweight_runner'


@pytest.mark.small
class DescribeGremlinSessionLightweightRunner:
    def it_defaults_lightweight_runner_to_on(self) -> None:
        session = GremlinSession()

        assert session.lightweight_runner is True

    def it_accepts_lightweight_runner_off(self) -> None:
        session = GremlinSession(lightweight_runner=False)

        assert session.lightweight_runner is False


@pytest.mark.small
class DescribeExtractTomlFieldsLightweightRunner:
    def it_surfaces_the_merged_lightweight_runner_value(self) -> None:
        merged = GremlinConfig(lightweight_runner=False)

        fields = _extract_toml_fields(merged)

        assert fields[6] is False
        assert fields[7] is None
