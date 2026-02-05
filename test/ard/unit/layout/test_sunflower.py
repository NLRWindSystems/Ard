from pathlib import Path

import numpy as np
import scipy.spatial
import openmdao.api as om

import pytest
import windIO

import ard.utils.test_utils as test_utils
import ard.layout.sunflower as sunflower


class TestSunflowerFarm:

    def setup_method(self):

        self.N_turbines = 25
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
        }

        self.model = om.Group()
        self.sunflower = self.model.add_subsystem(
            "sunflower",
            sunflower.SunflowerFarmLayout(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling_options has what we need for the layout
        assert "modeling_options" in [k for k, _ in self.sunflower.options.items()]

        assert "layout" in self.sunflower.options["modeling_options"].keys()
        assert (
            "N_turbines" in self.sunflower.options["modeling_options"]["layout"].keys()
        )

        assert "wind_farm" in self.modeling_options["windIO_plant"].keys()
        assert "turbine" in self.modeling_options["windIO_plant"]["wind_farm"].keys()
        assert (
            "rotor_diameter"
            in self.modeling_options["windIO_plant"]["wind_farm"]["turbine"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.sunflower.list_inputs()]
            for var_to_check in [
                "spacing_target",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.sunflower.list_outputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "spacing_effective_primary",
                "spacing_effective_secondary",
            ]:
                assert var_to_check in output_list

    def test_compute_pyrite_7D(self):

        # check out the layout at some target spacing
        spacing = 7.0
        self.prob.set_val("sunflower.spacing_target", spacing)

        # check out the spacing at some target spacing
        self.prob.set_val("sunflower.spacing_target", spacing)

        # run the model
        self.prob.run_model()

        # make sure the effective spacings get values
        assert np.isclose(
            self.prob.get_val("spacing_effective_primary"), self.D_rotor * spacing
        )  # machine prec by definition
        assert np.isclose(
            self.prob.get_val("spacing_effective_secondary"), self.D_rotor * spacing
        )  # machine prec by definition

        # collect data to validate
        validation_data = {
            "x_turbines": self.prob.get_val("x_turbines", units="km"),
            "y_turbines": self.prob.get_val("y_turbines", units="km"),
        }

        # validate data against pyrite file
        test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_sunflower_7D_pyrite.npz",
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )

    def test_compute_pyrite_4D(self):

        # check out the layout at some target spacing
        spacing = 4.0
        self.prob.set_val("sunflower.spacing_target", spacing)

        # run the model
        self.prob.run_model()

        # make sure the effective spacings get values
        assert np.isclose(
            self.prob.get_val("spacing_effective_primary"), self.D_rotor * spacing
        )  # machine prec by definition
        assert np.isclose(
            self.prob.get_val("spacing_effective_secondary"), self.D_rotor * spacing
        )  # machine prec by definition

        # collect data to validate
        validation_data = {
            "x_turbines": self.prob.get_val("x_turbines", units="km"),
            "y_turbines": self.prob.get_val("y_turbines", units="km"),
        }

        # validate data against pyrite file
        test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_sunflower_4D_pyrite.npz",
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )

    def test_target_spacing(self):

        for spacing in [4.0, 7.0]:

            # run the model
            self.prob.set_val("sunflower.spacing_target", spacing)
            self.prob.run_model()

            # get the turbine locations and their co-distance matrices
            points = np.vstack(
                [
                    self.prob.get_val("x_turbines", units="km"),
                    self.prob.get_val("y_turbines", units="km"),
                ]
            ).T
            dist_mtx = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(points)
            )
            np.fill_diagonal(dist_mtx, np.inf)  # self-distance not meaningful, remove
            d_mean_NN = np.mean(np.min(dist_mtx, axis=0))

            # target spacing should set mean nearest-neighbor distance
            assert np.isclose(self.D_rotor * spacing / 1e3, d_mean_NN)
