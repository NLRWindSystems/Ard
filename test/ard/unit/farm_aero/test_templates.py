import numpy as np
import openmdao.api as om

import floris

import pytest

import ard.wind_query as wq
import ard.farm_aero.templates as templates


class TestFarmAeroTemplate:

    def setup_method(self):

        self.N_turbines = 4
        self.D_rotor = 130.0

        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    },
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
            },
            "aero": {
                "return_turbine_output": True,
            },
        }

        self.model = om.Group()
        self.fa_temp = self.model.add_subsystem(
            "fa_temp",
            templates.FarmAeroTemplate(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        """
        make sure the modeling_options has what we need for farmaero
        """

        assert "modeling_options" in [k for k, _ in self.fa_temp.options.items()]

        assert "layout" in self.fa_temp.options["modeling_options"].keys()
        assert "N_turbines" in self.fa_temp.options["modeling_options"]["layout"].keys()

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the inputs in the component match what we planned
            input_list = [k for k, v in self.fa_temp.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "yaw_turbines",
            ]:
                assert var_to_check in input_list

    def test_compute(self):

        # make sure that an attempt to compute on the un-specialized class fails
        with pytest.raises(NotImplementedError):
            x_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            y_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            yaw_turbines = np.zeros_like(x_turbines)
            self.prob.set_val("x_turbines", x_turbines)
            self.prob.set_val("y_turbines", y_turbines)
            self.prob.set_val("yaw_turbines", yaw_turbines)
            self.prob.run_model()


class TestBatchFarmPowerTemplate:

    def setup_method(self):
        self.wq = wq.WindQuery(
            np.array([0.0, 180.0, 360.0]),
            np.array([1.0, 10.0, 30.0]),
            0.06,
        )

        self.N_turbines = 4
        self.D_rotor = 130.0

        self.modeling_options = {
            "windIO_plant": {
                "site": {
                    "energy_resource": {
                        "wind_resource": {
                            "wind_direction": self.wq.get_directions().tolist(),
                            "wind_speed": self.wq.get_speeds().tolist(),
                            "turbulence_intensity": self.wq.get_TIs().tolist(),
                            "time": np.zeros_like(self.wq.get_speeds().tolist()),
                        },
                    },
                },
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    },
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
            },
            "aero": {
                "return_turbine_output": True,
            },
        }

        self.model = om.Group()
        self.bfp_temp = self.model.add_subsystem(
            "bfp_temp",
            templates.BatchFarmPowerTemplate(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        "make sure the modeling_options has what we need for farmaero"
        assert "modeling_options" in [k for k, _ in self.bfp_temp.options.items()]

        assert "layout" in self.bfp_temp.options["modeling_options"].keys()
        assert (
            "N_turbines" in self.bfp_temp.options["modeling_options"]["layout"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the inputs in the component match what we planned
            input_list = [k for k, v in self.bfp_temp.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "yaw_turbines",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.bfp_temp.list_outputs()]
            for var_to_check in [
                "power_farm",
                "power_turbines",
                "thrust_turbines",
            ]:
                assert var_to_check in output_list

    def test_compute(self):

        # make sure that an attempt to compute on the un-specialized class fails
        with pytest.raises(NotImplementedError):
            x_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            y_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            yaw_turbines = np.zeros_like(x_turbines)
            self.prob.set_val("x_turbines", x_turbines)
            self.prob.set_val("y_turbines", y_turbines)
            self.prob.set_val("yaw_turbines", yaw_turbines)
            self.prob.run_model()


class TestFarmAEPTemplate:

    def setup_method(self):
        self.wr = floris.WindRose(
            wind_directions=np.array([270, 280]),
            wind_speeds=np.array([6.0, 7.0, 8.0]),
            ti_table=0.06,
        )

        self.N_turbines = 4
        self.D_rotor = 130.0

        self.modeling_options = {
            "windIO_plant": {
                "site": {
                    "energy_resource": {
                        "wind_resource": {
                            "name": "unit test resource",
                            "wind_direction": self.wr.wind_directions.tolist(),
                            "wind_speed": self.wr.wind_speeds.tolist(),
                            "probability": {
                                "data": self.wr.freq_table.tolist(),
                                "dims": ["wind_direction", "wind_speed"],
                            },
                            "turbulence_intensity": {
                                "data": self.wr.ti_table.tolist(),
                                "dims": ["wind_direction", "wind_speed"],
                            },
                        },
                    },
                },
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    },
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
            },
            "aero": {
                "return_turbine_output": True,
            },
        }

        self.model = om.Group()
        self.aep_temp = self.model.add_subsystem(
            "bfp_temp",
            templates.FarmAEPTemplate(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling_options has what we need for farmaero
        assert "modeling_options" in [k for k, _ in self.aep_temp.options.items()]

        assert "layout" in self.aep_temp.options["modeling_options"].keys()
        assert (
            "N_turbines" in self.aep_temp.options["modeling_options"]["layout"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.aep_temp.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "yaw_turbines",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.aep_temp.list_outputs()]
            for var_to_check in [
                "AEP_farm",
                "power_farm",
                "power_turbines",
                "thrust_turbines",
            ]:
                assert var_to_check in output_list

    def test_compute(self):

        # make sure that an attempt to compute on the un-specialized class fails
        with pytest.raises(NotImplementedError):
            x_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            y_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            yaw_turbines = np.zeros_like(x_turbines)
            self.prob.set_val("x_turbines", x_turbines)
            self.prob.set_val("y_turbines", y_turbines)
            self.prob.set_val("yaw_turbines", yaw_turbines)
            self.prob.run_model()
