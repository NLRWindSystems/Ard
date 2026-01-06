import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om

import pytest

import ard.viz.plot_layout as plot_layout


@pytest.mark.usefixtures("subtests")
class TestPlotLayout:

    def setup_method(self):
        self.modeling_options = {
            "farm": {
                "N_turbines": 25,
            },
        }

        self.model = om.Group()
        self.aep_temp = self.model.add_subsystem(
            "vizplot",
            plot_layout.OutputLayout(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self, subtests):
        # make sure the modeling_options has what we need for farmaero
        with subtests.test("modeling_options"):
            assert "modeling_options" in [k for k, _ in self.aep_temp.options.items()]

        with subtests.test("farm"):
            assert "farm" in self.aep_temp.options["modeling_options"].keys()

        with subtests.test("N_turbines"):
            assert (
                "N_turbines" in self.aep_temp.options["modeling_options"]["farm"].keys()
            )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.aep_temp.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
            ]:

                with subtests.test(f"input {var_to_check}"):
                    assert var_to_check in input_list

            output_list = [k for k, v in self.aep_temp.list_outputs()]

            with subtests.test(f"outputs don't exist yet"):
                assert len(output_list) == 0  # no outputs should exist

    def test_compute(self, subtests):
        # make sure no errors
        x_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        y_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        X, Y = np.meshgrid(x_turbines, y_turbines)
        self.prob.set_val("x_turbines", X.flatten())
        self.prob.set_val("y_turbines", Y.flatten())

        self.prob.run_model()

        with subtests.test("figure generated"):
            assert len(plt.get_fignums()) > 0  # matplotlib should create a figure

        plt.close("all")
        with subtests.test("figure closed"):
            assert len(plt.get_fignums()) == 0  # matplotlib should destroy all figures
