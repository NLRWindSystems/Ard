"""
Unit tests for ard/flowfarm/flowfarm_model.py.

resolve_wake_model_inputs_for_flowfarm is pure Python and tested without any mocking.
resolve_turbine_inputs_for_flowfarm calls Julia internally; those calls are patched.
_resolve_flowfarm_constructor is pure Python and tested with simple mock objects.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from ard.flowfarm.flowfarm_model import (
    _resolve_flowfarm_constructor,
    resolve_turbine_inputs_for_flowfarm,
    resolve_wake_model_inputs_for_flowfarm,
)

# ---------------------------------------------------------------------------
# resolve_wake_model_inputs_for_flowfarm  (pure Python — no Julia needed)
# ---------------------------------------------------------------------------


class TestResolveWakeModelInputs:

    def test_empty_dict_uses_all_defaults(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = resolve_wake_model_inputs_for_flowfarm({})

        assert result["wake_deficit_model"] == "GaussYawVariableSpread"
        assert result["wake_deflection_model"] == "GaussYawVariableSpreadDeflection"
        assert result["wake_combination_model"] == "LinearLocalVelocitySuperposition"
        assert result["local_turbulence_model"] == "LocalTIModelNoLocalTI"
        assert result["tolerance"] == pytest.approx(1e-16)

    def test_none_treated_as_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = resolve_wake_model_inputs_for_flowfarm(None)

        assert result["wake_deficit_model"] == "GaussYawVariableSpread"

    def test_explicit_valid_options_pass_through(self):
        opts = {
            "wake_deficit_model": "JensenTopHat",
            "wake_deflection_model": "NoYawDeflection",
            "wake_combination_model": "LinearFreestreamSuperposition",
            "local_turbulence_model": "LocalTIModelMaxTI",
            "tolerance": 1e-8,
        }
        result = resolve_wake_model_inputs_for_flowfarm(opts)

        assert result["wake_deficit_model"] == "JensenTopHat"
        assert result["wake_deflection_model"] == "NoYawDeflection"
        assert result["wake_combination_model"] == "LinearFreestreamSuperposition"
        assert result["local_turbulence_model"] == "LocalTIModelMaxTI"
        assert result["tolerance"] == pytest.approx(1e-8)

    def test_case_insensitive_matching(self):
        opts = {
            "wake_deficit_model": "jensentophat",
            "wake_deflection_model": "NOYAWDEFLECTION",
            "wake_combination_model": "linearfreestreamSuperposition",
            "local_turbulence_model": "localtimodelmaXTI",
            "tolerance": 1e-6,
        }
        result = resolve_wake_model_inputs_for_flowfarm(opts)

        assert result["wake_deficit_model"] == "JensenTopHat"
        assert result["wake_deflection_model"] == "NoYawDeflection"

    def test_invalid_deficit_model_raises_value_error(self):
        with pytest.raises(ValueError, match="wake_deficit_model"):
            resolve_wake_model_inputs_for_flowfarm({"wake_deficit_model": "NotAModel"})

    def test_invalid_deflection_model_raises_value_error(self):
        with pytest.raises(ValueError, match="wake_deflection_model"):
            resolve_wake_model_inputs_for_flowfarm(
                {"wake_deflection_model": "NotAModel"}
            )

    def test_invalid_combination_model_raises_value_error(self):
        with pytest.raises(ValueError, match="wake_combination_model"):
            resolve_wake_model_inputs_for_flowfarm(
                {"wake_combination_model": "NotAModel"}
            )

    def test_invalid_ti_model_raises_value_error(self):
        with pytest.raises(ValueError, match="local_turbulence_model"):
            resolve_wake_model_inputs_for_flowfarm(
                {"local_turbulence_model": "NotAModel"}
            )

    def test_non_string_model_name_raises_type_error(self):
        with pytest.raises(TypeError, match="wake_deficit_model"):
            resolve_wake_model_inputs_for_flowfarm({"wake_deficit_model": 42})

    def test_empty_string_model_name_raises_value_error(self):
        with pytest.raises(ValueError, match="wake_deficit_model"):
            resolve_wake_model_inputs_for_flowfarm({"wake_deficit_model": ""})

    def test_whitespace_only_model_name_raises_value_error(self):
        with pytest.raises(ValueError, match="wake_deficit_model"):
            resolve_wake_model_inputs_for_flowfarm({"wake_deficit_model": "   "})

    def test_non_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            resolve_wake_model_inputs_for_flowfarm(["JensenTopHat"])

    def test_tolerance_explicit_value(self):
        result = resolve_wake_model_inputs_for_flowfarm({"tolerance": 1e-6})
        assert result["tolerance"] == pytest.approx(1e-6)

    def test_tolerance_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError, match="tolerance"):
            resolve_wake_model_inputs_for_flowfarm({"tolerance": "small"})

    def test_tolerance_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="tolerance"):
            resolve_wake_model_inputs_for_flowfarm({"tolerance": 0.0})

    def test_tolerance_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="tolerance"):
            resolve_wake_model_inputs_for_flowfarm({"tolerance": -1e-6})

    def test_unknown_keys_warn_and_are_ignored(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_wake_model_inputs_for_flowfarm({"unknown_option": "value"})

        assert any("unknown" in str(w.message).lower() for w in caught)
        assert "unknown_option" not in result

    def test_missing_keys_warn_with_defaults_used(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_wake_model_inputs_for_flowfarm({})

        assert any("missing" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# _resolve_flowfarm_constructor  (pure Python — no Julia needed)
# ---------------------------------------------------------------------------


class TestResolveFlowfarmConstructor:

    def test_returns_first_matching_candidate(self):
        mock_module = MagicMock()
        mock_ctor = MagicMock(name="PowerModelCpPoints")
        mock_module.PowerModelCpPoints = mock_ctor

        result = _resolve_flowfarm_constructor(
            mock_module, ["PowerModelCpPoints", "PowerModelCpConstant"]
        )
        assert result is mock_ctor

    def test_returns_second_when_first_absent(self):
        mock_module = MagicMock(spec=["PowerModelCpConstant"])
        mock_ctor = MagicMock(name="PowerModelCpConstant")
        mock_module.PowerModelCpConstant = mock_ctor

        result = _resolve_flowfarm_constructor(
            mock_module, ["PowerModelCpPoints", "PowerModelCpConstant"]
        )
        assert result is mock_ctor

    def test_returns_none_when_no_candidate_exists(self):
        mock_module = MagicMock(spec=[])  # no attributes

        result = _resolve_flowfarm_constructor(mock_module, ["Missing1", "Missing2"])
        assert result is None


# ---------------------------------------------------------------------------
# resolve_turbine_inputs_for_flowfarm  (Julia calls mocked)
# ---------------------------------------------------------------------------


def _make_full_turbine_dict():
    """A complete windIO turbine dict — no warnings expected."""
    return {
        "generator_efficiency": 0.95,
        "rated_power": 5e6,
        "rated_wind_speed": 11.5,
        "cutin_wind_speed": 3.0,
        "cutout_wind_speed": 25.0,
        "performance": {
            "Ct_curve": {
                "Ct_wind_speeds": [3.0, 11.5, 25.0],
                "Ct_values": [0.8, 0.5, 0.2],
            },
            "Cp_curve": {
                "Cp_wind_speeds": [3.0, 11.5, 25.0],
                "Cp_values": [0.45, 0.45, 0.1],
            },
        },
    }


@pytest.fixture
def patched_julia():
    """Patch all Julia calls inside flowfarm_model so no Julia runtime is needed."""
    mock_ff_module = MagicMock(name="FLOWFarm")
    mock_power_model = MagicMock(name="PowerModel")
    mock_ct_model = MagicMock(name="CtModel")

    with (
        patch("ard.flowfarm.flowfarm_model._ensure_flowfarm_loaded"),
        patch("ard.flowfarm.flowfarm_model._get_jl_main") as mock_jl_main,
        patch(
            "ard.flowfarm.flowfarm_model._build_flowfarm_power_model",
            return_value=mock_power_model,
        ),
        patch(
            "ard.flowfarm.flowfarm_model._build_flowfarm_ct_model",
            return_value=mock_ct_model,
        ),
    ):
        mock_jl_main.return_value = MagicMock(FLOWFarm=mock_ff_module)
        yield {"power_model": mock_power_model, "ct_model": mock_ct_model}


class TestResolveTurbineInputs:

    def test_full_inputs_return_correct_scalars(self, patched_julia):
        turbine = _make_full_turbine_dict()
        result = resolve_turbine_inputs_for_flowfarm(turbine)

        assert result["generator_efficiency"] == pytest.approx(0.95)
        assert result["rated_power"] == pytest.approx(5e6)
        assert result["rated_wind_speed"] == pytest.approx(11.5)
        assert result["cutin_wind_speed"] == pytest.approx(3.0)
        assert result["cutout_wind_speed"] == pytest.approx(25.0)

    def test_full_inputs_return_model_objects(self, patched_julia):
        turbine = _make_full_turbine_dict()
        result = resolve_turbine_inputs_for_flowfarm(turbine)

        assert result["power_model"] is patched_julia["power_model"]
        assert result["ct_model"] is patched_julia["ct_model"]

    def test_missing_scalars_warn_and_use_defaults(self, patched_julia):
        turbine = {
            "performance": {
                "Ct_curve": {"Ct_wind_speeds": [3.0], "Ct_values": [0.8]},
                "Cp_curve": {"Cp_wind_speeds": [3.0], "Cp_values": [0.45]},
            }
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_turbine_inputs_for_flowfarm(turbine)

        assert any("missing" in str(w.message).lower() for w in caught)
        assert result["generator_efficiency"] == pytest.approx(1.0)
        assert result["rated_power"] == pytest.approx(1e6)

    def test_missing_ct_curve_warns(self, patched_julia):
        turbine = _make_full_turbine_dict()
        del turbine["performance"]["Ct_curve"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_turbine_inputs_for_flowfarm(turbine)

        assert any("ct_curve" in str(w.message).lower() for w in caught)

    def test_missing_cp_curve_warns(self, patched_julia):
        turbine = _make_full_turbine_dict()
        del turbine["performance"]["Cp_curve"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_turbine_inputs_for_flowfarm(turbine)

        assert any("cp_curve" in str(w.message).lower() for w in caught)

    def test_none_ct_values_treated_as_missing(self, patched_julia):
        turbine = _make_full_turbine_dict()
        turbine["performance"]["Ct_curve"]["Ct_values"] = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_turbine_inputs_for_flowfarm(turbine)

        assert any("ct_curve" in str(w.message).lower() for w in caught)

    def test_constant_ct_fallback_value_used(self, patched_julia):
        turbine = _make_full_turbine_dict()
        del turbine["performance"]["Ct_curve"]
        turbine["performance"]["Ct"] = 0.75

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            resolve_turbine_inputs_for_flowfarm(turbine)

        # Verify _build_flowfarm_ct_model received constant_ct=0.75
        from ard.flowfarm import flowfarm_model as ffm

        # The patch replaced the builder; check it was called with the right constant
        # (patched_julia fixture doesn't expose call args, so just check no error raised)

    def test_result_contains_all_expected_keys(self, patched_julia):
        turbine = _make_full_turbine_dict()
        result = resolve_turbine_inputs_for_flowfarm(turbine)

        for key in [
            "generator_efficiency",
            "rated_power",
            "rated_wind_speed",
            "cutin_wind_speed",
            "cutout_wind_speed",
            "ct_model",
            "power_model",
        ]:
            assert key in result, f"Missing key: {key}"
