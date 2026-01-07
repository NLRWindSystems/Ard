from pathlib import Path

import numpy as np

import openmdao.api as om
import openmdao.utils.assert_utils as om_utils

from ard.utils.io import load_yaml  # we grab a yaml loader here
from ard.eco.eagle_density import EagleDensityFunction

import pytest


@pytest.mark.usefixtures("subtests")
class TestEagleDensityFunction:

    def setup_method(self):

        # load input
        path_inputs = Path(__file__).parent.absolute() / "inputs"
        input_dict = load_yaml(path_inputs / "ard_system_eagle_density.yaml")
        modeling_options = self.modeling_options = input_dict["modeling_options"]

        # create the OpenMDAO model
        model = om.Group()
        self.collection = model.add_subsystem(
            "eagle_density",
            EagleDensityFunction(modeling_options=modeling_options),
            promotes=["*"],
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_eagledensity(self, subtests):

        Rmax = 500.0  # m
        R = lambda x, y: np.sqrt(x * x + y * y)
        # THETA = lambda x, y: np.atan2(x, y)
        F = lambda x, y: -R(x, y) * R(x, y) / (Rmax * Rmax) + 2 * R(x, y) / Rmax

        # generated from casting random bytestreams to int
        seeds = [2005908367, 1273391448, 2557384174, 2195599068, 1604240584]

        for seed in seeds:

            # create a repeatable rng
            rng = np.random.default_rng(seed)

            # evaluate the turbines in the grid
            x_turbines = rng.uniform(
                -1000.0, 1000.0, self.modeling_options["layout"]["N_turbines"]
            )
            y_turbines = rng.uniform(
                -1000.0, 1000.0, self.modeling_options["layout"]["N_turbines"]
            )
            F_turbines_reference = F(x_turbines, y_turbines)

            # set up the model to run
            self.prob.set_val("x_turbines", x_turbines, units="m")
            self.prob.set_val("y_turbines", y_turbines, units="m")
            self.prob.run_model()

            # assert that the values are close to the reference
            with subtests.test(f"eagle_normalized_density check seed {seed}"):
                assert np.allclose(
                    self.prob.get_val("eagle_normalized_density"),
                    F_turbines_reference,
                    atol=5.0e-2,
                )

    def test_gradient_eagledensity(self, subtests):

        Rmax = 500.0  # m
        R = lambda x, y: np.sqrt(x * x + y * y)
        # THETA = lambda x, y: np.atan2(x, y)
        F = lambda x, y: -R(x, y) * R(x, y) / (Rmax * Rmax) + 2 * R(x, y) / Rmax

        # generated from casting random bytestreams to int
        seeds = [2005908367, 1273391448, 2557384174, 2195599068, 1604240584]

        for seed in seeds:

            # create a repeatable rng
            rng = np.random.default_rng(seed)

            # evaluate the turbines in the grid
            x_turbines = rng.uniform(
                -1000.0, 1000.0, self.modeling_options["layout"]["N_turbines"]
            )
            y_turbines = rng.uniform(
                -1000.0, 1000.0, self.modeling_options["layout"]["N_turbines"]
            )
            F_turbines_reference = F(x_turbines, y_turbines)

            # set up the model to run
            self.prob.set_val("x_turbines", x_turbines, units="m")
            self.prob.set_val("y_turbines", y_turbines, units="m")
            self.prob.run_model()

            # check the partial derivatives
            with subtests.test(f"eagle_normalized_density gradient check seed {seed}"):
                partials = self.prob.check_partials(
                    method="fd",
                    step=1.0e-8,
                    out_stream=None,
                )
                om_utils.assert_check_partials(partials)
