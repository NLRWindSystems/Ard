"""
Unit tests for ard/farm_aero/flowfarm/_jl_bootstrap.py.

juliacall is mocked entirely via sys.modules — Julia does not need to be installed
for these tests.
"""

import sys
from unittest.mock import MagicMock

import pytest

import ard.flowfarm._jl_bootstrap as bootstrap


@pytest.fixture(autouse=True)
def reset_bootstrap_globals(monkeypatch):
    """Reset module-level singletons before each test so state never leaks."""
    monkeypatch.setattr(bootstrap, "_jl_runtime", None)
    monkeypatch.setattr(bootstrap, "_flowfarm_env_initialized", False)


@pytest.fixture
def mock_juliacall(monkeypatch):
    """Inject a fake juliacall module so Julia is never started."""
    mock = MagicMock(name="juliacall")
    mock.Main = MagicMock(name="Main")
    mock.Pkg = MagicMock(name="Pkg")
    monkeypatch.setitem(sys.modules, "juliacall", mock)
    return mock


class TestGetJuliaRuntime:

    def test_returns_main_and_pkg(self, mock_juliacall):
        jl_main, jl_pkg = bootstrap.get_julia_runtime()
        assert jl_main is mock_juliacall.Main
        assert jl_pkg is mock_juliacall.Pkg

    def test_singleton_returns_same_tuple(self, mock_juliacall):
        result1 = bootstrap.get_julia_runtime()
        result2 = bootstrap.get_julia_runtime()
        assert result1 is result2


class TestEnsureFlowfarmLoaded:

    def test_activates_instantiates_and_loads_flowfarm(self, mock_juliacall):
        result = bootstrap.ensure_flowfarm_loaded()

        mock_juliacall.Pkg.activate.assert_called_once()
        mock_juliacall.Pkg.instantiate.assert_called_once()
        mock_juliacall.Main.seval.assert_called_once_with("using FLOWFarm")
        assert result is mock_juliacall.Main

    def test_does_not_reinitialize_on_second_call(self, mock_juliacall):
        bootstrap.ensure_flowfarm_loaded()
        mock_juliacall.Main.FLOWFarm = MagicMock(name="FLOWFarm")
        mock_juliacall.Pkg.activate.reset_mock()
        mock_juliacall.Pkg.instantiate.reset_mock()
        mock_juliacall.Main.seval.reset_mock()

        bootstrap.ensure_flowfarm_loaded()

        mock_juliacall.Pkg.activate.assert_not_called()
        mock_juliacall.Pkg.instantiate.assert_not_called()
        mock_juliacall.Main.seval.assert_not_called()
