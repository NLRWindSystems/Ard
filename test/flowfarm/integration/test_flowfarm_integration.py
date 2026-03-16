"""
Integration tests for the FLOWFarm integration in Ard.

These tests require Julia and FLOWFarm to be installed.
They are marked @pytest.mark.julia and will be automatically skipped
(with a printed note) when Julia is not available.

Run only these tests:
    pytest -m julia test/flowfarm/integration

Run without these tests:
    pytest -m "not julia" ...
"""
from pathlib import Path

import numpy as np
import openmdao.api as om
import pytest
import yaml

import ard
import ard.utils.test_utils


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_PATH_TURBINE = (
    Path(ard.__file__).parents[1]
    / "examples"
    / "data"
    / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
)

_N_TURBINES = 9  # 3x3 grid — small enough for fast integration tests
_ROTOR_DIAMETER = 130.0
_SPACING = 5.0  # rotor diameters


def _grid_layout(n_side, spacing_d, rotor_d):
    coords = spacing_d * rotor_d * np.arange(n_side)
    X, Y = np.meshgrid(coords, coords)
    return X.flatten(), Y.flatten()


def _load_turbine_yaml():
    with open(_PATH_TURBINE) as f:
        return yaml.safe_load(f)


def _make_aep_modeling_options():
    import floris

    turbine = _load_turbine_yaml()
    n_side = 3
    directions = np.linspace(0.0, 360.0, 9, endpoint=False)
    speeds = np.array([6.0, 8.0, 10.0, 12.0])
    wind_rose = floris.WindRose(
        wind_directions=directions,
        wind_speeds=speeds,
        ti_table=0.06,
    )
    return {
        "windIO_plant": {
            "wind_farm": {"name": "integration test farm", "turbine": turbine},
            "site": {
                "energy_resource": {
                    "wind_resource": {
                        "wind_direction": wind_rose.wind_directions.tolist(),
                        "wind_speed": wind_rose.wind_speeds.tolist(),
                        "probability": {
                            "data": wind_rose.freq_table.tolist(),
                            "dim": ["wind_direction", "wind_speed"],
                        },
                        "turbulence_intensity": {
                            "data": wind_rose.ti_table.tolist(),
                            "dim": ["wind_direction", "wind_speed"],
                        },
                        "shear": 0.2,
                        "reference_height": 110.0,
                    }
                }
            },
        },
        "layout": {"N_turbines": n_side ** 2},
        "aero": {"return_turbine_output": True},
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.julia
class TestFlowFarmBootstrap:

    def test_ensure_flowfarm_loaded_returns_main(self):
        from ard.flowfarm._jl_bootstrap import ensure_flowfarm_loaded

        jl_main = ensure_flowfarm_loaded()
        assert jl_main is not None

    def test_flowfarm_module_accessible_after_load(self):
        from ard.flowfarm._jl_bootstrap import ensure_flowfarm_loaded

        jl_main = ensure_flowfarm_loaded()
        assert hasattr(jl_main, "FLOWFarm")


# ---------------------------------------------------------------------------
# FLOWFarmAEP component
# ---------------------------------------------------------------------------


@pytest.mark.julia
class TestFLOWFarmAEPIntegration:

    def setup_method(self):
        from ard.farm_aero.flowfarm import FLOWFarmAEP

        modeling_options = _make_aep_modeling_options()
        model = om.Group()
        self.component = model.add_subsystem(
            "aepFLOWFarm",
            FLOWFarmAEP(modeling_options=modeling_options),
        )
        self.prob = om.Problem(model)
        self.prob.setup()

        n_side = 3
        X, Y = _grid_layout(n_side, _SPACING, _ROTOR_DIAMETER)
        self.X = X
        self.Y = Y
        self.prob.set_val("aepFLOWFarm.x_turbines", X)
        self.prob.set_val("aepFLOWFarm.y_turbines", Y)
        self.prob.set_val("aepFLOWFarm.yaw_turbines", np.zeros(len(X)))

    def test_inputs_declared(self):
        input_list = [k for k, _ in self.component.list_inputs(val=False)]
        for var in ["x_turbines", "y_turbines", "yaw_turbines"]:
            assert var in input_list

    def test_outputs_declared(self):
        output_list = [k for k, _ in self.component.list_outputs(val=False)]
        for var in ["AEP_farm", "power_farm"]:
            assert var in output_list

    def test_compute_returns_positive_aep(self):
        self.prob.run_model()
        aep = self.prob.get_val("aepFLOWFarm.AEP_farm")
        assert float(aep) > 0.0

    def test_compute_aep_consistent_on_repeated_calls(self):
        self.prob.run_model()
        aep1 = float(self.prob.get_val("aepFLOWFarm.AEP_farm"))
        self.prob.run_model()
        aep2 = float(self.prob.get_val("aepFLOWFarm.AEP_farm"))
        assert aep1 == pytest.approx(aep2, rel=1e-10)

    def test_partials_check(self):
        """Analytical gradients should agree with finite differences to 1%."""
        self.prob.run_model()
        data = self.prob.check_totals(
            of=["aepFLOWFarm.AEP_farm"],
            wrt=["aepFLOWFarm.x_turbines", "aepFLOWFarm.y_turbines"],
            method="fd",
            compact_print=True,
        )
        for key, vals in data.items():
            rel_err = vals.get("rel error")
            if rel_err is not None:
                assert abs(rel_err.forward) < 0.01, (
                    f"Partial derivative rel error too large for {key}: {rel_err.forward:.4f}"
                )

    def test_aep_decreases_with_closer_spacing(self):
        """AEP should be lower for a tighter layout due to increased wake losses."""
        self.prob.run_model()
        aep_spread = float(self.prob.get_val("aepFLOWFarm.AEP_farm"))

        n_side = 3
        X_tight, Y_tight = _grid_layout(n_side, 2.0, _ROTOR_DIAMETER)  # 2D spacing
        self.prob.set_val("aepFLOWFarm.x_turbines", X_tight)
        self.prob.set_val("aepFLOWFarm.y_turbines", Y_tight)
        self.prob.run_model()
        aep_tight = float(self.prob.get_val("aepFLOWFarm.AEP_farm"))

        assert aep_tight < aep_spread


# ---------------------------------------------------------------------------
# Pyrite regression  (generated on first run with rewrite=True)
# ---------------------------------------------------------------------------


@pytest.mark.julia
class TestFLOWFarmAEPPyrite:

    def setup_method(self):
        from ard.farm_aero.flowfarm import FLOWFarmAEP

        modeling_options = _make_aep_modeling_options()
        model = om.Group()
        model.add_subsystem("aepFLOWFarm", FLOWFarmAEP(modeling_options=modeling_options))
        prob = om.Problem(model)
        prob.setup()

        n_side = 3
        X, Y = _grid_layout(n_side, _SPACING, _ROTOR_DIAMETER)
        prob.set_val("aepFLOWFarm.x_turbines", X)
        prob.set_val("aepFLOWFarm.y_turbines", Y)
        prob.set_val("aepFLOWFarm.yaw_turbines", np.zeros(len(X)))
        prob.run_model()
        self.prob = prob

    def test_aep_pyrite(self, subtests):
        test_data = {
            "aep_farm": self.prob.get_val("aepFLOWFarm.AEP_farm", units="GW*h"),
            "power_farm": self.prob.get_val("aepFLOWFarm.power_farm", units="MW"),
        }
        pyrite_path = Path(__file__).parent / "test_flowfarm_aep_pyrite.npz"

        pyrite_data = ard.utils.test_utils.pyrite_validator(
            test_data,
            pyrite_path,
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to regenerate reference data
        )

        for key in test_data:
            with subtests.test(key):
                assert np.allclose(test_data[key], pyrite_data[key], rtol=5e-3)
