"""Verify the baseline remains untraced without pytest-cov."""

from unittest.mock import MagicMock, patch

import pytest

from pytest_gremlins.plugin import CoverageMode, GremlinSession, _set_session, pytest_sessionstart


@pytest.mark.small
class DescribePrivateCoverageMode:
    """PRIVATE mode collects line contexts in a later subprocess only."""

    def it_does_not_start_or_register_a_tracer_for_the_baseline(self) -> None:
        session = MagicMock(spec=pytest.Session)
        session.config.pluginmanager.get_plugin.return_value = None
        session.config.pluginmanager.register = MagicMock()
        _set_session(GremlinSession(enabled=True, coverage_mode=CoverageMode.PRIVATE))

        with patch('pytest_gremlins.plugin.coverage.Coverage') as coverage_factory:
            pytest_sessionstart(session)

        coverage_factory.assert_not_called()
        session.config.pluginmanager.register.assert_not_called()

    def it_leaves_disabled_sessions_untraced(self) -> None:
        session = MagicMock(spec=pytest.Session)
        _set_session(GremlinSession(enabled=False, coverage_mode=CoverageMode.PRIVATE))

        with patch('pytest_gremlins.plugin.coverage.Coverage') as coverage_factory:
            pytest_sessionstart(session)

        coverage_factory.assert_not_called()
