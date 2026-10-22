from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import yaml

import numpy as np
import openmdao.api as om

import floris
import pytest
import ard.utils.io
import ard.utils.test_utils
import ard.wind_query as wq
import ard.farm_aero.floris as farmaero_floris


class TestFLORISSurrogateDELs:

    def setup_method(self):
        self.FLORIS = farmaero_floris.FLORISSurrogateDELs()
        self.FLORIS.fmodel = Mock()

    def test_feature_order(self, monkeypatch):
        """Preserve condition/turbine order and convert TI and power to percent."""
        speeds = np.arange(16.0).reshape(2, 2, 4)
        intensities = (np.arange(16.0).reshape(2, 2, 4) + 1) / 100.0
        self.FLORIS.fmodel.get_turbine_sector_average_wind_speed.return_value = speeds
        self.FLORIS.fmodel.get_turbine_sector_average_TI.return_value = intensities
        self.FLORIS.fmodel.core.farm = SimpleNamespace(
            turbine_map=[
                SimpleNamespace(power_thrust_table={"power": [0.0, 2000.0]}),
                SimpleNamespace(power_thrust_table={"power": [0.0, 4000.0]}),
            ],
            power_setpoints=np.array([[2e6, 2e6], [0.5e6, 3e6]]),
            yaw_angles=np.array([[1.0, -2.0], [3.0, -4.0]]),
        )
        params = {"test": "parameters"}
        self.FLORIS.modeling_options = {"surrogate_loads": params}
        predictions = [np.full((4, 1), value) for value in range(1, 5)]
        predictors = [Mock(return_value=value) for value in predictions]
        factory = Mock(side_effect=predictors)
        monkeypatch.setattr(farmaero_floris.ard_slf, "ANN_DEL_factory", factory)

        result = self.FLORIS.get_raw_DELs()

        assert [call.args[0] for call in factory.call_args_list] == [
            "BladeRoot",
            "Shaft",
            "TowerBase",
            "YawBearings",
        ]
        for call in factory.call_args_list:
            assert call.args[1] is params
        expected = np.array(
            [
                [0, 1, 2, 3, 1, 2, 3, 4, 1, 100],
                [4, 5, 6, 7, 5, 6, 7, 8, -2, 50],
                [8, 9, 10, 11, 9, 10, 11, 12, 3, 25],
                [12, 13, 14, 15, 13, 14, 15, 16, -4, 75],
            ]
        )
        for predictor, actual, prediction in zip(predictors, result, predictions):
            predictor.assert_called_once()
            np.testing.assert_allclose(predictor.call_args.args[0], expected)
            np.testing.assert_array_equal(actual, prediction)

    def test_unlimited_setpoints_are_full_demand(self, monkeypatch):
        """Translate FLORIS unlimited setpoints to 100% surrogate demand."""
        self.FLORIS.fmodel.get_turbine_sector_average_wind_speed.return_value = (
            np.zeros((1, 1, 4))
        )
        self.FLORIS.fmodel.get_turbine_sector_average_TI.return_value = np.zeros(
            (1, 1, 4)
        )
        self.FLORIS.fmodel.core.farm = SimpleNamespace(
            turbine_map=[SimpleNamespace(power_thrust_table={"power": [0.0, 2000.0]})],
            power_setpoints=np.array([[np.inf]]),
            yaw_angles=np.array([[0.0]]),
        )
        self.FLORIS.modeling_options = {"surrogate_loads": {}}
        predictor = Mock(return_value=np.zeros((1, 1)))
        monkeypatch.setattr(
            farmaero_floris.ard_slf, "ANN_DEL_factory", Mock(return_value=predictor)
        )

        self.FLORIS.get_raw_DELs()

        np.testing.assert_array_equal(
            predictor.call_args.args[0][:, -1],
            [100.0],
        )

    def test_frequency_weighting(self):
        """Weight each condition's turbine block without changing raw predictions."""
        self.FLORIS.N_turbines = 2
        self.FLORIS.wind_query = SimpleNamespace(
            freq_table=np.array([[0.1, 0.0], [0.3, 0.6]]),
            non_zero_freq_mask=np.array([[True, False], [True, True]]),
        )
        raw = tuple(np.arange(1.0, 7.0).reshape(-1, 1) * scale for scale in range(1, 5))
        originals = tuple(value.copy() for value in raw)
        self.FLORIS.get_raw_DELs = Mock(return_value=raw)

        weighted = self.FLORIS.get_DELs()

        self.FLORIS.get_raw_DELs.assert_called_once_with()
        for actual, original, unchanged in zip(weighted, originals, raw):
            np.testing.assert_allclose(
                actual,
                original * np.array([0.1, 0.1, 0.3, 0.3, 0.6, 0.6])[:, None],
            )
            np.testing.assert_array_equal(unchanged, original)

    def test_compute(self, monkeypatch):
        """Publish all four summed DEL outputs after the parent aero computation."""
        parent_compute = Mock()
        monkeypatch.setattr(farmaero_floris.FLORISAEP, "compute", parent_compute)
        self.FLORIS.fmodel.get_farm_AEP.return_value = 1234.0
        self.FLORIS.get_DELs = Mock(
            return_value=tuple(
                np.array([[1.0], [2.0]]) * scale for scale in range(1, 5)
            )
        )
        inputs, outputs = {}, {}

        self.FLORIS.compute(inputs, outputs)

        parent_compute.assert_called_once_with(inputs, outputs)
        assert outputs == {
            "blade_root_DEL": 3.0,
            "shaft_DEL": 6.0,
            "tower_base_DEL": 9.0,
            "yaw_bearings_DEL": 12.0,
            "AEP_farm": 1234.0,
        }


class TestFLORISCurtailment:

    @pytest.fixture(autouse=True)
    def setup_problem(self, tmp_path, monkeypatch):
        """Set up a tiny farm with DEL models that return scaled power demand."""
        monkeypatch.chdir(tmp_path)
        first_weights = np.zeros((60, 10))
        first_weights[0, 9] = 1.0
        second_weights = np.zeros((1, 60))
        second_weights[0, 0] = 1.0
        names = ["BladeRoot", "Shaft", "TowerBase", "YawBearings"]
        params = {
            name: {
                "zScore_norm": {
                    "mu_x": np.zeros(10),
                    "sigma_x": np.ones(10),
                    "mu_t": 0.0,
                    "sigma_t": float(index),
                },
                "weights": {
                    "W1": first_weights,
                    "b1": np.zeros(60),
                    "W2": second_weights,
                    "b2": 0.0,
                },
            }
            for index, name in enumerate(names, start=1)
        }
        turbine_path = (
            Path(ard.__file__).parents[1]
            / "examples/data/windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        with open(turbine_path) as turbine_file:
            turbine = yaml.safe_load(turbine_file)
        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {"name": "unit test farm", "turbine": turbine},
                "site": {
                    "energy_resource": {
                        "wind_resource": {
                            "wind_direction": [270.0],
                            "wind_speed": [8.0, 12.0],
                            "probability": {"data": [[0.25, 0.75]]},
                            "turbulence_intensity": {"data": [[0.06, 0.08]]},
                            "reference_height": turbine["hub_height"],
                        }
                    }
                },
            },
            "layout": {"N_turbines": 2},
            "aero": {"return_turbine_output": True},
            "surrogate_loads": params,
        }
        self.prob = om.Problem(reports=False)
        self.FLORIS = self.prob.model.add_subsystem(
            "curtailment",
            farmaero_floris.FLORISCurtailment(
                modeling_options=self.modeling_options,
                case_title="unit test",
                data_path="",
            ),
        )
        self.prob.setup()
        self.prob.final_setup()
        self.prob.set_val("curtailment.x_turbines", [0.0, 1000.0])
        self.prob.set_val("curtailment.y_turbines", [0.0, 0.0])
        yield
        self.prob.cleanup()

    def test_setup(self):
        """Declare percent setpoints per condition and turbine and scalar DELs."""
        np.testing.assert_array_equal(
            self.prob.get_val(
                "curtailment.turbine_power_pct_setpoints", units="percent"
            ),
            np.full((2, 2), 100.0),
        )
        for name in [
            "blade_root_DEL",
            "shaft_DEL",
            "tower_base_DEL",
            "yaw_bearings_DEL",
        ]:
            np.testing.assert_array_equal(
                self.prob.get_val(f"curtailment.{name}", units="kN*m"), [0.0]
            )
        assert self.FLORIS.N_wind_conditions == 2

    @pytest.mark.parametrize(
        "setpoints",
        [
            np.full((2, 2), 100.0),
            np.array([[50.0, 75.0], [25.0, 100.0]]),
        ],
    )
    def test_compute(self, setpoints):
        """Run FLORIS and the ANNs with known setpoints and weighted DEL totals."""
        self.prob.set_val("curtailment.turbine_power_pct_setpoints", setpoints)

        self.prob.run_model()

        rated_power = (
            max(
                self.FLORIS.fmodel.core.farm.turbine_definitions[0][
                    "power_thrust_table"
                ]["power"]
            )
            * 1000.0
        )
        np.testing.assert_allclose(
            self.FLORIS.fmodel.core.farm.power_setpoints,
            setpoints / 100.0 * rated_power,
        )
        assert self.FLORIS.fmodel.core.wake.enable_turbine_turbulence_grid
        assert self.FLORIS.fmodel.get_operation_model() == "simple-derating"
        frequencies = np.array([[0.25], [0.75]])
        for scale, name in enumerate(
            [
                "blade_root_DEL",
                "shaft_DEL",
                "tower_base_DEL",
                "yaw_bearings_DEL",
            ],
            start=1,
        ):
            np.testing.assert_allclose(
                self.FLORIS.rDEL_outputs[scale - 1],
                (setpoints * scale).reshape(-1, 1),
            )
            np.testing.assert_allclose(
                self.FLORIS.tDEL_outputs[scale - 1],
                (setpoints * frequencies * scale).reshape(-1, 1),
            )
            np.testing.assert_allclose(
                self.prob.get_val(f"curtailment.{name}"),
                np.sum(setpoints * frequencies) * scale,
            )
        powers = self.prob.get_val("curtailment.power_turbines")
        assert powers.shape == (2, 2)
        assert np.all(powers >= 0.0)
        assert np.all(powers <= (setpoints / 100.0 * rated_power).T + 1e-6)
        np.testing.assert_allclose(
            self.prob.get_val("curtailment.power_farm"), powers.sum(axis=0)
        )
        np.testing.assert_allclose(
            self.prob.get_val("curtailment.AEP_farm"),
            np.sum(powers.sum(axis=0) * frequencies.flatten()) * 8760.0,
        )
        assert (self.FLORIS.dir_floris / "batch.yaml").is_file()

    def test_compute_with_zero_probability_bin(self):
        """Exclude zero-probability conditions from FLORIS setpoints and DELs."""
        original_wind_query = self.FLORIS.wind_query
        self.FLORIS.wind_query = floris.WindRose(
            wind_directions=original_wind_query.wind_directions,
            wind_speeds=original_wind_query.wind_speeds,
            freq_table=np.array([[0.0, 0.75]]),
            ti_table=original_wind_query.ti_table,
        )
        self.FLORIS.wind_query.reference_height = original_wind_query.reference_height
        setpoints = np.array([[50.0, 75.0], [25.0, 100.0]])
        self.prob.set_val("curtailment.turbine_power_pct_setpoints", setpoints)

        self.prob.run_model()

        rated_power = (
            max(
                self.FLORIS.fmodel.core.farm.turbine_definitions[0][
                    "power_thrust_table"
                ]["power"]
            )
            * 1000.0
        )
        np.testing.assert_allclose(
            self.FLORIS.fmodel.core.farm.power_setpoints,
            setpoints[1:] / 100.0 * rated_power,
        )
        assert self.FLORIS.rDEL_outputs[0].shape == (2, 1)
        np.testing.assert_allclose(
            self.FLORIS.tDEL_outputs[0], self.FLORIS.rDEL_outputs[0]
        )

    def test_rejects_peak_shaving(self):
        """Reject the incompatible peak-shaving option before running FLORIS."""
        self.modeling_options["floris"] = {"peak_shaving_fraction": 0.4}
        with pytest.raises(NotImplementedError, match="peak_shaving_fraction"):
            self.prob.run_model()


class TestFLORISFarmComponent:

    def setup_method(self):
        pass


class TestFLORISBatchPower:

    def setup_method(self):

        # create the wind query
        directions = np.linspace(0.0, 360.0, 21)
        speeds = np.linspace(0.0, 30.0, 21)[1:]
        WS, WD = np.meshgrid(speeds, directions)
        wind_query = wq.WindQuery(WD.flatten(), WS.flatten())
        wind_query.set_TI_using_IEC_method()

        # create the farm layout specification
        farm_spec = {}
        farm_spec["xD_farm"], farm_spec["yD_farm"] = [
            7 * v.flatten()
            for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        ]

        # set up the modeling options
        path_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        with open(path_turbine) as f_yaml:
            data_turbine_yaml = yaml.safe_load(f_yaml)
        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "name": "unit test farm",
                    "turbine": data_turbine_yaml,
                },
                "site": {
                    "energy_resource": {
                        "wind_resource": {
                            "wind_direction": wind_query.get_directions().tolist(),
                            "wind_speed": wind_query.get_speeds().tolist(),
                            "turbulence_intensity": wind_query.get_TIs().tolist(),
                            "time": np.zeros_like(wind_query.get_speeds().tolist()),
                            "shear": {
                                "alpha": 0.585,
                            },
                        },
                        "reference_height": 90.0,
                    },
                },
            },
            "layout": {
                "N_turbines": len(farm_spec["xD_farm"]),
            },
            "aero": {
                "return_turbine_output": True,
            },
            "floris": {
                "peak_shaving_fraction": 0.4,
                "peak_shaving_TI_threshold": 0.0,
            },
        }

        # create the OpenMDAO model
        model = om.Group()
        self.FLORIS = model.add_subsystem(
            "batchFLORIS",
            farmaero_floris.FLORISBatchPower(
                modeling_options=self.modeling_options,
                case_title="letsgo",
                data_path="",
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_setup(self):
        "make sure the modeling_options has what we need for farmaero"

        assert "case_title" in [k for k, _ in self.FLORIS.options.items()]
        assert "modeling_options" in [k for k, _ in self.FLORIS.options.items()]

        assert "layout" in self.FLORIS.options["modeling_options"].keys()
        assert "N_turbines" in self.FLORIS.options["modeling_options"]["layout"].keys()

        # make sure that the inputs in the component match what we planned
        input_list = [k for k, v in self.FLORIS.list_inputs(val=False)]
        for var_to_check in [
            "x_turbines",
            "y_turbines",
            "yaw_turbines",
        ]:
            assert var_to_check in input_list

        # make sure that the outputs in the component match what we planned
        output_list = [k for k, v in self.FLORIS.list_outputs(val=False)]
        for var_to_check in [
            "power_farm",
            "power_turbines",
            "thrust_turbines",
        ]:
            assert var_to_check in output_list

    def test_compute_pyrite(self):

        x_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        y_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        X, Y = [v.flatten() for v in np.meshgrid(x_turbines, y_turbines)]
        yaw_turbines = np.zeros_like(X)
        self.prob.set_val("batchFLORIS.x_turbines", X)
        self.prob.set_val("batchFLORIS.y_turbines", Y)
        self.prob.set_val("batchFLORIS.yaw_turbines", yaw_turbines)

        self.prob.run_model()

        # collect data to validate
        validation_data = {
            "power_farm": self.prob.get_val("batchFLORIS.power_farm", units="MW"),
            "power_turbines": self.prob.get_val(
                "batchFLORIS.power_turbines", units="MW"
            ),
            "thrust_turbines": self.prob.get_val(
                "batchFLORIS.thrust_turbines", units="kN"
            ),
        }

        # validate data against pyrite file
        ard.utils.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_floris_batch_pyrite.npz",
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )


class TestFLORISAEP:

    def setup_method(self):

        # create the wind query
        directions = np.linspace(0.0, 360.0, 21)
        speeds = np.linspace(0.0, 30.0, 21)[1:]
        wind_rose = floris.WindRose(
            wind_directions=directions,
            wind_speeds=speeds,
            ti_table=0.06,
        )

        # create the farm layout specification
        farm_spec = {}
        farm_spec["xD_farm"], farm_spec["yD_farm"] = [
            7 * v.flatten()
            for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        ]

        # set up the modeling options
        path_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        with open(path_turbine) as f_yaml:
            data_turbine_yaml = yaml.safe_load(f_yaml)
        modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "name": "unit test farm",
                    "turbine": data_turbine_yaml,
                },
                "site": {
                    "energy_resource": {
                        "wind_resource": {
                            "wind_direction": wind_rose.wind_directions.tolist(),
                            "wind_speed": wind_rose.wind_speeds.tolist(),
                            "probability": {
                                "data": wind_rose.freq_table.tolist(),
                                "dim": [
                                    "wind_direction",
                                    "wind_speed",
                                ],
                            },
                            "turbulence_intensity": {
                                "data": wind_rose.ti_table.tolist(),
                                "dim": [
                                    "wind_direction",
                                    "wind_speed",
                                ],
                            },
                            "shear": {
                                "alpha": 0.585,
                            },
                            "reference_height": 110.0,
                        },
                    },
                },
            },
            "layout": {
                "N_turbines": len(farm_spec["xD_farm"]),
            },
            "aero": {
                "return_turbine_output": True,
            },
            "floris": {
                "peak_shaving_fraction": 0.4,
                "peak_shaving_TI_threshold": 0.0,
            },
        }

        # create the OpenMDAO model
        model = om.Group()
        self.FLORIS = model.add_subsystem(
            "aepFLORIS",
            farmaero_floris.FLORISAEP(
                modeling_options=modeling_options,
                case_title="letsgo",
                data_path="",
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_setup(self):
        "make sure the modeling_options has what we need for farmaero"
        assert "case_title" in [k for k, _ in self.FLORIS.options.items()]
        assert "modeling_options" in [k for k, _ in self.FLORIS.options.items()]

        assert "layout" in self.FLORIS.options["modeling_options"].keys()
        assert "N_turbines" in self.FLORIS.options["modeling_options"]["layout"].keys()

        # make sure that the inputs in the component match what we planned
        input_list = [k for k, v in self.FLORIS.list_inputs(val=False)]
        for var_to_check in [
            "x_turbines",
            "y_turbines",
            "yaw_turbines",
        ]:
            assert var_to_check in input_list

        # make sure that the outputs in the component match what we planned
        output_list = [k for k, v in self.FLORIS.list_outputs(val=False)]
        for var_to_check in [
            "AEP_farm",
            "power_farm",
            "power_turbines",
            "thrust_turbines",
        ]:
            assert var_to_check in output_list

    def test_compute_pyrite(self, subtests):

        x_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        y_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        X, Y = [v.flatten() for v in np.meshgrid(x_turbines, y_turbines)]
        yaw_turbines = np.zeros_like(X)
        self.prob.set_val("aepFLORIS.x_turbines", X)
        self.prob.set_val("aepFLORIS.y_turbines", Y)
        self.prob.set_val("aepFLORIS.yaw_turbines", yaw_turbines)

        self.prob.run_model()

        # collect data to validate
        test_data = {
            "aep_farm": self.prob.get_val("aepFLORIS.AEP_farm", units="GW*h"),
            "power_farm": self.prob.get_val("aepFLORIS.power_farm", units="MW"),
            "power_turbines": self.prob.get_val("aepFLORIS.power_turbines", units="MW"),
            "thrust_turbines": self.prob.get_val(
                "aepFLORIS.thrust_turbines", units="kN"
            ),
        }
        # validate data against pyrite file
        pyrite_data = ard.utils.test_utils.pyrite_validator(
            test_data,
            Path(__file__).parent / "test_floris_aep_pyrite.npz",
            # rtol_val=5e-3, # this parameter sets the relative tolerance for validation checks. Uncomment if needed.
            # rewrite=True,  # uncomment to write new pyrite file
            load_only=True,
        )

        for key in test_data:
            with subtests.test(key):
                assert np.allclose(test_data[key], pyrite_data[key], rtol=5e-3)
