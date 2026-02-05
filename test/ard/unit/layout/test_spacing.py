import pytest
import numpy as np
import ard.layout.spacing
import openmdao.api as om


@pytest.mark.usefixtures("subtests")
class TestTurbineSpacingFunctions:
    def setup_method(self):
        pass

    def test_calculate_turbine_spacing(self):

        x_turbines = np.array([0.0, 0.0, 10.0])
        y_turbines = np.array([0.0, 10.0, 0.0])

        test_result = ard.layout.spacing.calculate_turbine_spacing(
            x_turbines, y_turbines
        )

        assert np.allclose(test_result, np.array([10, 10, 14.1421356]))

    def test_calculate_turbine_spacing_jac(self, subtests):

        x_turbines = np.array([0.0, 0.0, 10.0])
        y_turbines = np.array([0.0, 10.0, 0.0])

        test_result = ard.layout.spacing.calculate_turbine_spacing_jac(
            x_turbines, y_turbines
        )

        with subtests.test("derivatives of spacing wrt x turbine locations"):
            assert np.allclose(
                test_result[0],
                np.array(
                    [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -0.70710678, 0.70710678]]
                ),
            )


@pytest.mark.usefixtures("subtests")
class TestTurbineSpacingComponent:
    def setup_method(self):
        xt_in = np.array([10, 30])
        yt_in = np.array([10, 10])

        self.N_turbines = 2

        modeling_options = {
            "layout": {
                "N_turbines": self.N_turbines,
            },
            "platform": {
                "N_anchors": 4,
                "N_anchor_dimensions": 2,
                "min_mooring_line_length_m": 10000,
            },
        }

        prob = om.Problem(model=om.Group())
        prob.model.add_subsystem(
            "sc",
            ard.layout.spacing.TurbineSpacing(
                modeling_options=modeling_options,
            ),
            promotes=["*"],
        )

        prob.model.set_input_defaults("x_turbines", xt_in, units="km")
        prob.model.set_input_defaults("y_turbines", yt_in, units="km")
        prob.setup()
        prob.run_model()

        self.prob = prob

    def test_mooring_design_component_output(self):
        assert np.allclose(
            self.prob["turbine_spacing"],
            np.array([20.0]),
        )

    def test_mooring_design_component_output(self, subtests):
        totals = self.prob.compute_totals(
            of=["turbine_spacing"],
            wrt=["x_turbines", "y_turbines"],
        )

        totals_expected = {
            ("turbine_spacing", "x_turbines"): np.array([[-1.0, 1.0]]),
            ("turbine_spacing", "y_turbines"): np.array([[0.0, 0.0]]),
        }

        with subtests.test("spacing wrt x_turbines"):
            assert np.allclose(
                totals[("turbine_spacing", "x_turbines")],
                totals_expected[("turbine_spacing", "x_turbines")],
            )

        with subtests.test("spacing wrt y_turbines"):
            assert np.allclose(
                totals[("turbine_spacing", "y_turbines")],
                totals_expected[("turbine_spacing", "y_turbines")],
            )
