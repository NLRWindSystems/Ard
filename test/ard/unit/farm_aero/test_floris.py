from pathlib import Path

import yaml

import numpy as np
import openmdao.api as om

import floris
import pytest
import ard.utils.io
import ard.utils.test_utils
import ard.wind_query as wq
import ard.farm_aero.floris as farmaero_floris
import ard.farm_aero.templates as farmaero_templates


def _build_timeseries_windio(reference_height=None, h_ref=None):
    wind_resource = {
        "wind_direction": [270.0, 280.0],
        "wind_speed": [8.0, 9.0],
        "turbulence_intensity": [0.06, 0.06],
        "time": [0, 1],
    }
    if reference_height is not None:
        wind_resource["reference_height"] = reference_height
    if h_ref is not None:
        wind_resource["shear"] = {
            "h_ref": h_ref
        }

    return {
        "site": {
            "energy_resource": {
                "wind_resource": wind_resource,
            },
        },
    }


class TestWindResourceReferenceHeightAliases:

    def test_reference_height_or_h_ref_single_key(self):
        windio_ref = _build_timeseries_windio(reference_height=110.0)
        resource_ref = farmaero_templates.create_windresource_from_windIO(
            windio_ref, "timeseries"
        )
        assert resource_ref.reference_height == 110.0

        windio_href = _build_timeseries_windio(h_ref=95.0)
        resource_href = farmaero_templates.create_windresource_from_windIO(
            windio_href, "timeseries"
        )
        assert resource_href.reference_height == 95.0

    def test_reference_height_and_h_ref_same_value(self):
        windio = _build_timeseries_windio(reference_height=100.0, h_ref=100.0)
        resource = farmaero_templates.create_windresource_from_windIO(
            windio, "timeseries"
        )
        assert resource.reference_height == 100.0

    def test_reference_height_and_h_ref_different_values_raise(self):
        windio = _build_timeseries_windio(reference_height=100.0, h_ref=101.0)
        with pytest.raises(ValueError, match="reference_height.*h_ref"):
            farmaero_templates.create_windresource_from_windIO(windio, "timeseries")


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
