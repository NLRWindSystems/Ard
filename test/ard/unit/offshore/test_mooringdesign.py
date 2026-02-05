import pytest
import numpy as np
import jax
import jax.numpy as jnp
import ard.offshore.mooring_design_constant_depth as md
import openmdao.api as om
from jax.test_util import check_grads


@pytest.mark.usefixtures("subtests")
class TestMooringDesignSimpleFunctions:

    def setup_method(self):
        self.center = [10.0, 5.0]
        self.length = 10.0
        self.rotation_deg = [0.0]
        self.N = 4

    def test_generate_anchor_points(self, subtests):

        lines = md.generate_anchor_points(
            center=self.center,
            length=self.length,
            rotation_deg=self.rotation_deg[0],
            N=self.N,
        )

        with subtests.test("anchor 0"):
            assert np.allclose(lines[0], np.array([20.0, 5.0]))
        with subtests.test("anchor 1"):
            assert np.allclose(lines[1], np.array([10.0, 15.0]))
        with subtests.test("anchor 2"):
            assert np.allclose(lines[2], np.array([0.0, 5.0]))
        with subtests.test("anchor 3"):
            assert np.allclose(lines[3], np.array([10.0, -5.0]))

    def test_simple_mooring_design_1_turbine(self, subtests):

        x_anchors, y_anchors = md.simple_mooring_design(
            phi_platform=self.rotation_deg,
            x_turbines=np.array([self.center[0]]),
            y_turbines=np.array([self.center[1]]),
            length=self.length,
            N_turbines=1,
            N_anchors=self.N,
        )

        with subtests.test("anchor 0"):
            assert np.allclose(x_anchors, np.array([20.0, 10.0, 0.0, 10.0]))
        with subtests.test("anchor 1"):
            assert np.allclose(y_anchors, np.array([5.0, 15.0, 5.0, -5.0]))

    def test_simple_mooring_design_2_turbines(self, subtests):

        x_anchors, y_anchors = md.simple_mooring_design(
            phi_platform=np.array([self.rotation_deg[0], self.rotation_deg[0]]),
            x_turbines=np.array([self.center[0], self.center[0] + 2 * self.length]),
            y_turbines=np.array([self.center[1], self.center[1]]),
            length=self.length,
            N_turbines=2,
            N_anchors=self.N,
        )

        with subtests.test("x anchors"):
            assert np.allclose(
                x_anchors, np.array([[20.0, 10.0, 0.0, 10.0], [40.0, 30.0, 20.0, 30.0]])
            )
        with subtests.test("y anchors"):
            assert np.allclose(
                y_anchors, np.array([[5.0, 15.0, 5.0, -5.0], [5.0, 15.0, 5.0, -5.0]])
            )


class TestMooringDesignSimple3Turbines3Anchors2D:
    def setup_method(self):
        xt_in = np.array([10, 30])
        yt_in = np.array([10, 10])
        phi = np.array([0.0, 0.0])

        modeling_options = {
            "layout": {"N_turbines": 2},
            "platform": {
                "N_anchors": 4,
                "N_anchor_dimensions": 2,
                "min_mooring_line_length_m": 10000,
            },
        }

        prob = om.Problem(model=om.Group())
        prob.model.add_subsystem(
            "md",
            md.ConstantDepthMooringDesign(
                modeling_options=modeling_options, wind_query=None, bathymetry_data=None
            ),
            promotes=["*"],
        )

        prob.model.set_input_defaults("x_turbines", xt_in, units="km")
        prob.model.set_input_defaults("y_turbines", yt_in, units="km")
        prob.model.set_input_defaults("phi_platform", phi, units="deg")
        prob.setup()
        prob.run_model()

        self.prob0 = prob

    def test_mooring_design_component_output(self):
        assert np.allclose(
            self.prob0["x_anchors"],
            np.array([[20.0, 10.0, 0.0, 10.0], [40.0, 30.0, 20.0, 30.0]]),
        )
