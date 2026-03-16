"""
Unit tests for ard/farm_aero/flowfarm.py.

Julia is mocked entirely — these tests cover the Python-layer logic of
FLOWFarmComponent, FLOWFarmAEP, and FLOWFarmBatchPower without starting Julia.
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ard.farm_aero.flowfarm import FLOWFarmAEP, FLOWFarmBatchPower, FLOWFarmComponent
import ard.farm_aero.templates as templates


# ---------------------------------------------------------------------------
# _build_design_vector  (pure numpy — no Julia)
# ---------------------------------------------------------------------------


class TestBuildDesignVector:

    def _make_component(self):
        """Create a bare FLOWFarmComponent instance without calling __init__."""
        return FLOWFarmComponent.__new__(FLOWFarmComponent)

    def test_concatenates_x_y_yaw_in_order(self):
        comp = self._make_component()
        inputs = {
            "x_turbines": np.array([100.0, 200.0, 300.0]),
            "y_turbines": np.array([0.0, 50.0, 100.0]),
            "yaw_turbines": np.array([5.0, -5.0, 0.0]),
        }
        result = comp._build_design_vector(inputs)
        expected = np.array([100.0, 200.0, 300.0, 0.0, 50.0, 100.0, 5.0, -5.0, 0.0])
        assert np.allclose(result, expected)

    def test_returns_flat_array(self):
        comp = self._make_component()
        inputs = {
            "x_turbines": np.array([[1.0], [2.0]]),  # 2D input
            "y_turbines": np.array([[3.0], [4.0]]),
            "yaw_turbines": np.array([[0.0], [0.0]]),
        }
        result = comp._build_design_vector(inputs)
        assert result.ndim == 1
        assert len(result) == 6

    def test_accepts_list_inputs(self):
        comp = self._make_component()
        inputs = {
            "x_turbines": [10.0, 20.0],
            "y_turbines": [30.0, 40.0],
            "yaw_turbines": [0.0, 0.0],
        }
        result = comp._build_design_vector(inputs)
        assert np.allclose(result, [10.0, 20.0, 30.0, 40.0, 0.0, 0.0])

    def test_length_is_three_times_n_turbines(self):
        comp = self._make_component()
        n = 10
        inputs = {
            "x_turbines": np.zeros(n),
            "y_turbines": np.zeros(n),
            "yaw_turbines": np.zeros(n),
        }
        result = comp._build_design_vector(inputs)
        assert len(result) == 3 * n


# ---------------------------------------------------------------------------
# _evaluate_sparse / _evaluate_farm  caching logic
# ---------------------------------------------------------------------------


def _make_component_with_mock_julia(n_turbines=3):
    """Return a FLOWFarmComponent wired up with mock Julia objects."""
    comp = FLOWFarmComponent.__new__(FLOWFarmComponent)
    comp.N_turbines = n_turbines

    mock_jl = MagicMock(name="jl_main")
    comp._jl = mock_jl
    comp.flowfarm_module = MagicMock(name="FLOWFarm")
    comp.sparse_farm = MagicMock(name="sparse_farm")
    comp.sparse_struct = MagicMock(name="sparse_struct")
    comp.farm = MagicMock(name="farm")

    # Set up the Julia function return values
    grad_fn = getattr(comp.flowfarm_module, "calculate_aep_gradient!")
    grad_fn.return_value = (100.0, np.array([0.1] * (3 * n_turbines)))

    aep_fn = getattr(comp.flowfarm_module, "calculate_aep!")
    aep_fn.return_value = 100.0

    return comp


class TestEvaluateSparseCache:

    def test_caches_result_on_same_x(self):
        comp = _make_component_with_mock_julia()
        grad_fn = getattr(comp.flowfarm_module, "calculate_aep_gradient!")

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0])
        comp._evaluate_sparse(x)
        comp._evaluate_sparse(x)  # same x — should hit cache

        assert grad_fn.call_count == 1

    def test_reruns_on_different_x(self):
        comp = _make_component_with_mock_julia()
        grad_fn = getattr(comp.flowfarm_module, "calculate_aep_gradient!")

        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0])
        x2 = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.0, 0.0, 0.0])
        comp._evaluate_sparse(x1)
        comp._evaluate_sparse(x2)

        assert grad_fn.call_count == 2

    def test_stores_aep_and_grad_after_evaluation(self):
        n = 3
        comp = _make_component_with_mock_julia(n_turbines=n)
        grad_fn = getattr(comp.flowfarm_module, "calculate_aep_gradient!")
        mock_grad = np.arange(3 * n, dtype=float)
        grad_fn.return_value = (42.0, mock_grad)

        x = np.zeros(3 * n)
        comp._evaluate_sparse(x)

        assert comp._cached_sparse_aep == pytest.approx(42.0)
        assert np.allclose(comp._cached_sparse_grad, mock_grad)


class TestEvaluateFarmCache:

    def test_caches_result_on_same_x(self):
        comp = _make_component_with_mock_julia()
        aep_fn = getattr(comp.flowfarm_module, "calculate_aep!")

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0])
        comp._evaluate_farm(x)
        comp._evaluate_farm(x)

        assert aep_fn.call_count == 1

    def test_reruns_on_different_x(self):
        comp = _make_component_with_mock_julia()
        aep_fn = getattr(comp.flowfarm_module, "calculate_aep!")

        x1 = np.zeros(9)
        x2 = np.ones(9)
        comp._evaluate_farm(x1)
        comp._evaluate_farm(x2)

        assert aep_fn.call_count == 2

    def test_stores_aep_after_evaluation(self):
        comp = _make_component_with_mock_julia()
        aep_fn = getattr(comp.flowfarm_module, "calculate_aep!")
        aep_fn.return_value = 99.5

        comp._evaluate_farm(np.zeros(9))

        assert comp._cached_farm_aep == pytest.approx(99.5)


# ---------------------------------------------------------------------------
# _compute_aep_partials  gradient slicing
# ---------------------------------------------------------------------------


class TestComputeAEPPartials:

    def test_partials_sliced_correctly(self):
        n = 4
        comp = _make_component_with_mock_julia(n_turbines=n)
        grad = np.arange(3 * n, dtype=float)
        getattr(comp.flowfarm_module, "calculate_aep_gradient!").return_value = (
            1.0,
            grad,
        )

        inputs = {
            "x_turbines": np.zeros(n),
            "y_turbines": np.zeros(n),
            "yaw_turbines": np.zeros(n),
        }
        partials = {}
        comp._compute_aep_partials(inputs, partials)

        assert np.allclose(partials["AEP_farm", "x_turbines"], grad[:n])
        assert np.allclose(partials["AEP_farm", "y_turbines"], grad[n : 2 * n])
        assert np.allclose(partials["AEP_farm", "yaw_turbines"], grad[2 * n : 3 * n])


# ---------------------------------------------------------------------------
# Class hierarchy checks
# ---------------------------------------------------------------------------


class TestClassHierarchy:

    def test_flowfarm_aep_inherits_from_farm_aep_template(self):
        assert issubclass(FLOWFarmAEP, templates.FarmAEPTemplate)

    def test_flowfarm_aep_inherits_from_flowfarm_component(self):
        assert issubclass(FLOWFarmAEP, FLOWFarmComponent)

    def test_flowfarm_batch_power_inherits_from_batch_template(self):
        assert issubclass(FLOWFarmBatchPower, templates.BatchFarmPowerTemplate)

    def test_flowfarm_batch_power_inherits_from_flowfarm_component(self):
        assert issubclass(FLOWFarmBatchPower, FLOWFarmComponent)

    def test_flowfarm_aep_has_setup_partials(self):
        assert callable(getattr(FLOWFarmAEP, "setup_partials", None))

    def test_flowfarm_aep_has_compute(self):
        assert callable(getattr(FLOWFarmAEP, "compute", None))

    def test_flowfarm_aep_has_compute_partials(self):
        assert callable(getattr(FLOWFarmAEP, "compute_partials", None))

    def test_flowfarm_batch_power_has_compute(self):
        assert callable(getattr(FLOWFarmBatchPower, "compute", None))

    def test_flowfarm_batch_power_has_compute_partials(self):
        assert callable(getattr(FLOWFarmBatchPower, "compute_partials", None))
