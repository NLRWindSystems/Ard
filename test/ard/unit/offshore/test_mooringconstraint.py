import pytest
import numpy as np
import jax
import jax.numpy as jnp
import ard.offshore.mooring_constraint as mc
import openmdao.api as om
from jax.test_util import check_grads


class TestMooringConstraint3Turbines3Anchors2D:
    def setup_method(self):
        xt_in = np.array([10, 30, 50])
        yt_in = np.array([10, 10, 10])
        xa_in = np.array([[5, 10, 15], [25, 30, 35], [45, 50, 55]])
        ya_in = np.array([[5, 15, 5], [5, 15, 5], [5, 15, 5]])
        modeling_options = {
            "layout": {"N_turbines": 3},
            "platform": {"N_anchors": 3, "N_anchor_dimensions": 2},
        }

        prob = om.Problem(model=om.Group())
        prob.model.add_subsystem(
            "mc",
            mc.MooringConstraint(modeling_options=modeling_options),
            promotes=["*"],
        )

        prob.model.set_input_defaults("x_turbines", xt_in, units="km")
        prob.model.set_input_defaults("y_turbines", yt_in, units="km")
        prob.model.set_input_defaults("x_anchors", xa_in, units="km")
        prob.model.set_input_defaults("y_anchors", ya_in, units="km")
        prob.setup()
        prob.run_model()

        self.prob0 = prob

    def test_mooring_constraint_component_output(self):
        assert np.all(
            self.prob0["mooring_spacing"]
            == pytest.approx(np.array([10.0, 30.0, 10.0]), rel=1e-3)
        )


class TestMooringConstraint3Turbines3Anchors3D:
    def setup_method(self):
        xt_in = np.array([10, 30, 50])
        yt_in = np.array([10, 10, 10])
        xa_in = np.array([[5, 10, 15], [25, 30, 35], [45, 50, 55]])
        ya_in = np.array([[5, 15, 5], [5, 15, 5], [5, 15, 5]])
        za_in = np.array([[-5, -5, -5], [-5, -5, -5], [-5, -5, -5]])
        modeling_options = {
            "layout": {"N_turbines": 3},
            "platform": {"N_anchors": 3, "N_anchor_dimensions": 3},
        }

        prob = om.Problem(model=om.Group())
        prob.model.add_subsystem(
            "mc",
            mc.MooringConstraint(modeling_options=modeling_options),
            promotes=["*"],
        )

        prob.model.set_input_defaults("x_turbines", xt_in, units="km")
        prob.model.set_input_defaults("y_turbines", yt_in, units="km")
        prob.model.set_input_defaults("x_anchors", xa_in, units="km")
        prob.model.set_input_defaults("y_anchors", ya_in, units="km")
        prob.model.set_input_defaults("z_anchors", za_in, units="km")
        prob.setup()
        prob.run_model()

        self.prob0 = prob

    def test_mooring_constraint_component_output(self):
        assert np.all(
            self.prob0["mooring_spacing"]
            == pytest.approx(np.array([10.0, 30.0, 10.0]), rel=1e-3)
        )


class TestMooringConstraint2Turbines1Anchors2D:
    def setup_method(self):
        xt_in1 = np.array([0, 20])
        yt_in1 = np.array([0, 0])
        xa_in1 = np.array([[-3, 3], [17, 23]])
        ya_in1 = np.array([[0, 0], [0, 0]])
        modeling_options1 = {
            "layout": {"N_turbines": 2},
            "platform": {"N_anchors": 2, "N_anchor_dimensions": 2},
        }

        prob1 = om.Problem(model=om.Group())
        prob1.model.add_subsystem(
            "mc",
            mc.MooringConstraint(modeling_options=modeling_options1),
            promotes=["*"],
        )

        prob1.model.set_input_defaults("x_turbines", xt_in1, units="km")
        prob1.model.set_input_defaults("y_turbines", yt_in1, units="km")
        prob1.model.set_input_defaults("x_anchors", xa_in1, units="km")
        prob1.model.set_input_defaults("y_anchors", ya_in1, units="km")
        prob1.setup()
        prob1.run_model()
        totals1 = prob1.compute_totals(
            of=["mooring_spacing"],
            wrt=["x_turbines", "y_turbines", "x_anchors", "y_anchors"],
        )

        totals_expected1 = {
            ("mooring_spacing", "x_turbines"): np.array([[0.0, 0.0]]),
            ("mooring_spacing", "y_turbines"): np.array([[0.0, 0.0]]),
            ("mooring_spacing", "x_anchors"): np.array([[0.0, -1.0, 1.0, 0.0]]),
            ("mooring_spacing", "y_anchors"): np.array([[0.0, 0.0, 0.0, 0.0]]),
        }

        self.prob1 = prob1
        self.totals1 = totals1
        self.totals_expected1 = totals_expected1

    def test_mooring_constraint_component_output(self):
        assert np.all(
            self.prob1["mooring_spacing"] == pytest.approx(np.array([14.0]), rel=1e-3)
        )

    def test_mooring_constraint_component_jacobian0(self):
        assert np.all(
            self.totals1[("mooring_spacing", "x_turbines")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "x_turbines")])
        )

    def test_mooring_constraint_component_jacobian1(self):
        assert np.all(
            self.totals1[("mooring_spacing", "y_turbines")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "y_turbines")])
        )

    def test_mooring_constraint_component_jacobian2(self):
        assert np.all(
            self.totals1[("mooring_spacing", "x_anchors")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "x_anchors")])
        )

    def test_mooring_constraint_component_jacobian3(self):
        assert np.all(
            self.totals1[("mooring_spacing", "y_anchors")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "y_anchors")])
        )


class TestMooringConstraint2Turbines1Anchors3D:
    def setup_method(self):
        xt_in1 = np.array([0, 20])
        yt_in1 = np.array([0, 0])
        xa_in1 = np.array([[-3, 3], [17, 23]])
        ya_in1 = np.array([[0, 0], [0, 0]])
        za_in1 = np.array([[-10, -50], [-25, -10]])
        modeling_options1 = {
            "layout": {"N_turbines": 2},
            "platform": {"N_anchors": 2, "N_anchor_dimensions": 3},
        }

        prob1 = om.Problem(model=om.Group())
        prob1.model.add_subsystem(
            "mc",
            mc.MooringConstraint(modeling_options=modeling_options1),
            promotes=["*"],
        )

        prob1.model.set_input_defaults("x_turbines", xt_in1, units="km")
        prob1.model.set_input_defaults("y_turbines", yt_in1, units="km")
        prob1.model.set_input_defaults("x_anchors", xa_in1, units="km")
        prob1.model.set_input_defaults("y_anchors", ya_in1, units="km")
        prob1.model.set_input_defaults("z_anchors", za_in1, units="km")
        prob1.setup()
        prob1.run_model()
        totals1 = prob1.compute_totals(
            of=["mooring_spacing"],
            wrt=["x_turbines", "y_turbines", "x_anchors", "y_anchors", "z_anchors"],
        )

        totals_expected1 = {
            ("mooring_spacing", "x_turbines"): np.array([[-0.48060241, 0.0]]),
            ("mooring_spacing", "y_turbines"): np.array([[0.0, 0.0]]),
            ("mooring_spacing", "x_anchors"): np.array(
                [[0.0, -0.51760243, 0.9982048, 0.0]]
            ),
            ("mooring_spacing", "y_anchors"): np.array([[0.0, 0.0, 0.0, 0.0]]),
            ("mooring_spacing", "z_anchors"): np.array(
                [[0.0, -0.03105615, 0.05989229, 0.0]]
            ),
        }

        self.prob1 = prob1
        self.totals1 = totals1
        self.totals_expected1 = totals_expected1

    def test_mooring_constraint_component_output(self):
        assert np.all(
            self.prob1["mooring_spacing"] == pytest.approx(np.array([15.47]), rel=1e-3)
        )

    def test_mooring_constraint_component_jacobian0(self):
        assert np.all(
            self.totals1[("mooring_spacing", "x_turbines")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "x_turbines")])
        )

    def test_mooring_constraint_component_jacobian1(self):
        assert np.all(
            self.totals1[("mooring_spacing", "y_turbines")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "y_turbines")])
        )

    def test_mooring_constraint_component_jacobian2(self):
        assert np.all(
            self.totals1[("mooring_spacing", "x_anchors")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "x_anchors")])
        )

    def test_mooring_constraint_component_jacobian3(self):
        assert np.all(
            self.totals1[("mooring_spacing", "y_anchors")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "y_anchors")])
        )

    def test_mooring_constraint_component_jacobian4(self):
        assert np.all(
            self.totals1[("mooring_spacing", "z_anchors")]
            == pytest.approx(self.totals_expected1[("mooring_spacing", "z_anchors")])
        )


class TestMooringConstraintXY:
    def setup_method(self):
        pass

    def test_mooring_constraint_xy(self):
        xt_in = np.array([10, 30, 50])
        yt_in = np.array([10, 10, 10])
        xa_in = np.array([[5, 10, 15], [25, 30, 35], [45, 50, 55]])
        ya_in = np.array([[5, 15, 5], [5, 15, 5], [5, 15, 5]])

        test_result = mc.mooring_constraint_xy(xt_in, yt_in, xa_in, ya_in)

        assert np.all(
            test_result == pytest.approx(np.array([10.0, 30.0, 10.0]), rel=1e-3)
        )

    def test_mooring_constraint_xy_grad(self):
        xt_in = jnp.array([10, 30, 50], dtype=float)
        yt_in = jnp.array([10, 10, 10], dtype=float)
        xa_in = jnp.array([[5, 10, 15], [25, 30, 35], [45, 50, 55]], dtype=float)
        ya_in = jnp.array([[5, 15, 5], [5, 15, 5], [5, 15, 5]], dtype=float)

        try:
            check_grads(
                mc.mooring_constraint_xy,
                (xt_in, yt_in, xa_in, ya_in),
                order=1,
                modes="fwd",
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestMooringConstraintXYZ:
    def setup_method(self):
        pass

    def test_mooring_constraint_xyz(self):
        xt_in = np.array([10, 30, 50])
        yt_in = np.array([10, 10, 10])
        xa_in = np.array([[5, 10, 15], [25, 30, 35], [45, 50, 55]])
        ya_in = np.array([[5, 15, 5], [5, 15, 5], [5, 15, 5]])
        za_in = np.array([[-5, -5, -5], [-5, -5, -5], [-5, -5, -5]])

        test_result = mc.mooring_constraint_xyz(xt_in, yt_in, xa_in, ya_in, za_in)

        assert np.all(
            test_result == pytest.approx(np.array([10.0, 30.0, 10.0]), rel=1e-3)
        )

    def test_mooring_constraint_xyz_grad(self):
        xt_in = jnp.array([10, 30, 50], dtype=float)
        yt_in = jnp.array([10, 10, 10], dtype=float)
        xa_in = jnp.array([[5, 10, 15], [25, 30, 35], [45, 50, 55]], dtype=float)
        ya_in = jnp.array([[5, 15, 5], [5, 15, 5], [5, 15, 5]], dtype=float)
        za_in = np.array([[-5, -4, -3], [-6, -7, -8], [-9, -10, -11]], dtype=float)

        try:
            check_grads(
                mc.mooring_constraint_xyz,
                (xt_in, yt_in, xa_in, ya_in, za_in),
                order=1,
                modes="fwd",
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestCalcMooringDistances:
    def setup_method(self):
        pass

    def test_calc_mooring_distances(self):
        P_moorings_A = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]], dtype=float)

        P_moorings_B = jnp.array([[30, 10], [25, 5], [30, 15], [35, 5]], dtype=float)

        P_moorings_C = jnp.array([[50, 10], [45, 5], [50, 15], [55, 5]], dtype=float)

        mooring_points = jnp.array([P_moorings_A, P_moorings_B, P_moorings_C])

        test_result = mc.calc_mooring_distances(mooring_points)

        assert np.all(
            test_result == pytest.approx(np.array([10.0, 30.0, 10.0]), rel=1e-3)
        )

    def test_calc_mooring_distances_grad(self):
        P_moorings_A = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]], dtype=float)

        P_moorings_B = jnp.array([[30, 10], [25, 5], [30, 15], [35, 5]], dtype=float)

        P_moorings_C = jnp.array([[50, 10], [45, 5], [50, 15], [55, 5]], dtype=float)

        mooring_points = jnp.array([[P_moorings_A, P_moorings_B, P_moorings_C]])

        try:
            check_grads(
                mc.calc_mooring_distances, (mooring_points), order=1, modes="fwd"
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestConvertInputs_X_Y_To_XY:

    def setup_method(self):
        self.distance_point_to_mooring_grad = jax.grad(mc.convert_inputs_x_y_to_xy, [0])
        pass

    def test_convert_inputs_x_y_to_xy(self):

        xt_in = np.array([1, 10])
        yt_in = np.array([2, 20])
        xa_in = np.array([[3, 30, 300], [4, 40, 400]])
        ya_in = np.array([[5, 50, 500], [6, 60, 600]])

        pm_out_expected = np.array(
            [
                [[1, 2], [3, 5], [30, 50], [300, 500]],
                [[10, 20], [4, 6], [40, 60], [400, 600]],
            ],
            dtype=float,
        )

        pm_out = mc.convert_inputs_x_y_to_xy(xt_in, yt_in, xa_in, ya_in)

        assert np.all(pm_out == pm_out_expected)

    def test_convert_inputs_x_y_to_xy_grad(self):

        xt_in = jnp.array([1, 10], dtype=float)
        yt_in = jnp.array([2, 20], dtype=float)
        xa_in = jnp.array([[3, 30, 300], [4, 40, 400]], dtype=float)
        ya_in = jnp.array([[5, 50, 500], [6, 60, 600]], dtype=float)

        try:
            check_grads(
                mc.convert_inputs_x_y_to_xy, (xt_in, yt_in, xa_in, ya_in), order=1
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestConvertInputs_X_Y_Z_To_XYZ:

    def setup_method(self):
        self.distance_point_to_mooring_grad = jax.grad(
            mc.convert_inputs_x_y_z_to_xyz, [0]
        )
        pass

    def test_convert_inputs_x_y_z_to_xyz(self):

        xt_in = np.array([1, 10], dtype=float)
        yt_in = np.array([2, 20], dtype=float)
        zt_in = np.array([7, 70], dtype=float)
        xa_in = np.array([[3, 30, 300], [4, 40, 400]], dtype=float)
        ya_in = np.array([[5, 50, 500], [6, 60, 600]], dtype=float)
        za_in = np.array([[8, 80, 800], [9, 90, 900]], dtype=float)

        pm_out_expected = np.array(
            [
                [[1, 2, 7], [3, 5, 8], [30, 50, 80], [300, 500, 800]],
                [[10, 20, 70], [4, 6, 9], [40, 60, 90], [400, 600, 900]],
            ],
            dtype=float,
        )

        pm_out = mc.convert_inputs_x_y_z_to_xyz(
            xt_in, yt_in, zt_in, xa_in, ya_in, za_in
        )

        assert np.all(pm_out == pm_out_expected)

    def test_convert_inputs_x_y_z_to_xyz_grad(self):

        xt_in = jnp.array([1, 10], dtype=float)
        yt_in = jnp.array([2, 20], dtype=float)
        zt_in = jnp.array([7, 70], dtype=float)
        xa_in = jnp.array([[3, 30, 300], [4, 40, 400]], dtype=float)
        ya_in = jnp.array([[5, 50, 500], [6, 60, 600]], dtype=float)
        za_in = jnp.array([[8, 80, 800], [9, 90, 900]], dtype=float)

        try:
            check_grads(
                mc.convert_inputs_x_y_z_to_xyz,
                (xt_in, yt_in, zt_in, xa_in, ya_in, za_in),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestDistancePointToMooring:
    def setup_method(self):
        self.distance_point_to_mooring_grad = jax.grad(
            mc.distance_point_to_mooring, [0]
        )
        pass

    def test_distance_point_to_mooring_2d_near_end(self):

        point = jnp.array([0, 0])

        P_moorings = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(7.0710678118654755)

    def test_distance_point_to_mooring_2d_near_middle(self):

        point = jnp.array([7.0, 7.5])

        P_moorings = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(0.3535533905932738)

    def test_distance_point_to_mooring_3d_near_end_3d(self):

        point = jnp.array([0, 0, 0])

        P_moorings = jnp.array([[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(7.0710678118654755)

    def test_distance_point_to_mooring_3d_near_middle(self):

        point = jnp.array([7.0, 7.5, 0])

        P_moorings = jnp.array([[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(0.3535533905932738)

    def test_distance_point_to_mooring_2d_near_end_grad(self):

        point = jnp.array([0, 5], dtype=float)

        P_moorings = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]], dtype=float)

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)

        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0]))

    def test_distance_point_to_mooring_2d_near_middle_grad(self):

        point = jnp.array([9, 13], dtype=float)

        P_moorings = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]], dtype=float)

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)

        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0]))

    def test_distance_point_to_mooring_3d_near_end_grad(self):

        point = jnp.array([0, 5, 0], dtype=float)

        P_moorings = jnp.array(
            [[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]], dtype=float
        )

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)

        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0, 0.0]))

    def test_distance_point_to_mooring_3d_near_middle_grad(self):

        point = jnp.array([9, 13, 0], dtype=float)

        P_moorings = jnp.array(
            [[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]], dtype=float
        )

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)

        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0, 0.0]))


class TestDistanceMooringToMooring:
    def setup_method(self):
        self.distance_mooring_to_mooring_grad = jax.grad(
            mc.distance_mooring_to_mooring, [0]
        )
        self.distance_mooring_to_mooring_jac = jax.jacobian(
            mc.distance_mooring_to_mooring, [0]
        )
        pass

    def test_distance_mooring_to_mooring_2d_near_end(self):

        P_moorings_A = np.array([[10, 10], [5, 5], [10, 15], [15, 5]])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 20

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(10, rel=1e-3)

    def test_distance_mooring_to_mooring_2d_near_middle(self):

        P_moorings_A = np.array([[10, 10], [5, 5], [10, 15], [15, 5]])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 6
        P_moorings_B[:, 1] -= 10

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        assert float(test_result) == pytest.approx(1.0, rel=1e-2)

    def test_distance_mooring_to_mooring_2d_equal(self):

        P_moorings_A = np.array([[10, 10], [5, 5], [10, 15], [15, 5]])

        P_moorings_B = np.copy(P_moorings_A)

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(0.0, abs=1e-2)

    def test_distance_mooring_to_mooring_3d_near_end(self):

        P_moorings_A = np.array([[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 20

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(10, rel=1e-3)

    def test_distance_mooring_to_mooring_3d_near_middle(self):

        P_moorings_A = np.array([[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 6
        P_moorings_B[:, 1] -= 10

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        assert float(test_result) == pytest.approx(1.0, rel=1e-2)

    def test_distance_mooring_to_mooring_3d_equal(self):

        P_moorings_A = np.array([[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]])

        P_moorings_B = np.copy(P_moorings_A)

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(0.0, abs=1e-2)

    def test_distance_mooring_to_mooring_2d_near_end_grad(self):

        P_moorings_A = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]], dtype=float)

        P_moorings_B = jnp.array([[30, 10], [25, 5], [30, 15], [35, 5]], dtype=float)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
        )

    def test_distance_mooring_to_mooring_2d_near_middle_grad(self):

        P_moorings_A = jnp.array(
            [[100, 100], [50, 50], [100, 150], [150, 50]], dtype=float
        )

        P_moorings_B = jnp.array(
            [[160, 0], [110, -50], [160, 100], [210, -50]], dtype=float
        )

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
        )
        try:
            check_grads(
                mc.distance_mooring_to_mooring, (P_moorings_A, P_moorings_B), order=1
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )

    def test_distance_mooring_to_mooring_2d_equal_grad(self):

        P_moorings_A = jnp.array([[10, 10], [5, 5], [10, 15], [15, 5]], dtype=float)

        P_moorings_B = jnp.copy(P_moorings_A)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        )

    def test_distance_mooring_to_mooring_3d_near_end_grad(self):

        P_moorings_A = jnp.array(
            [[10, 10, 0], [5, 5, 0], [10, 15, 0], [15, 5, 0]], dtype=float
        )

        P_moorings_B = jnp.array(
            [[30, 10, 0], [25, 5, 0], [30, 15, 0], [35, 5, 0]], dtype=float
        )

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
            )
        )

    def test_distance_mooring_to_mooring_3d_near_middle_grad(self):

        P_moorings_A = jnp.array(
            [[10, 10, 0.0], [5, 5, 0.0], [10, 15, 0.0], [15, 5, 0.0]], dtype=float
        )

        P_moorings_B = jnp.array(
            [[16, 0, 0.0], [11, -5, 0.0], [16, 6, 0.0], [21, -5, 0.0]], dtype=float
        )

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
            )
        )

        try:
            check_grads(
                mc.distance_mooring_to_mooring, (P_moorings_A, P_moorings_B), order=1
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )

    def test_distance_mooring_to_mooring_3d_near_middle_end_grad(self):

        P_moorings_A = jnp.array(
            [[10, 10, 0.0], [5, 5, 0.0], [10, 15, 0.0], [15, 5, 0.0]], dtype=float
        )

        P_moorings_B = jnp.array(
            [[16, 0, 0.0], [11, -5, 0.0], [16, 5, 0.0], [21, -5, 0.0]], dtype=float
        )

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
            )
        )

    def test_distance_mooring_to_mooring_3d_equal_grad(self):

        P_moorings_A = jnp.array(
            [[10, 10, 0.0], [5, 5, 0.0], [10, 15, 0.0], [15, 5, 0.0]], dtype=float
        )

        P_moorings_B = jnp.copy(P_moorings_A)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        )
