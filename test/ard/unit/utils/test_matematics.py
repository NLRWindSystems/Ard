import numpy as np
import pytest
from jax import grad, test_util
import jax.numpy as jnp
import ard.utils.mathematics as math_utils


class TestSmoothMaxMin:
    def setup_method(self):
        self.smooth_max_grad = grad(math_utils.smooth_max)
        self.smooth_min_grad = grad(math_utils.smooth_min)
        pass

    def test_smooth_max_close(self):
        """
        Check that the smooth max function returns something greater
        than the true max when called with similar values.
        """

        test_list = np.array([0, 9.999, 10.00, 3.0])
        test_result = math_utils.smooth_max(x=test_list)

        assert test_result >= 10

    def test_smooth_max_not_close(self):
        """
        Check that the smooth max function returns the true max
        when called with very different values.
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = math_utils.smooth_max(x=test_list)

        assert test_result == 10

    def test_smooth_min_close(self):
        """
        Check that the smooth min function returns something less
        than the true min when called with similar values.
        """

        test_list = np.array([0, 9.999, 10.00, 3.0])
        test_result = math_utils.smooth_min(x=test_list)

        assert test_result <= 0.0

    def test_smooth_min_not_close(self):
        """
        Check that the smooth min function returns the true min
        when called with very different values.
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = math_utils.smooth_min(x=test_list)

        assert test_result == pytest.approx(0, rel=1e-15)

    def test_smooth_max_grad(self):
        """
        Check that the smooth min function is differentiable and the gradient
        is correct
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = self.smooth_max_grad(test_list)

        assert test_result == pytest.approx([0, 0, 1, 0], rel=1e-6)

        try:
            test_util.check_grads(math_utils.smooth_max, ([test_list]), order=1)
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )

    def test_smooth_min_grad(self):
        """
        Check that the smooth min function is differentiable and the gradient
        is correct
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = self.smooth_min_grad(test_list)

        assert test_result == pytest.approx([1, 0, 0, 0], rel=1e-6)


class TestSmoothNorm:
    def setup_method(self):
        self.smooth_norm_grad = grad(math_utils.smooth_norm, [0])
        self.norm_grad = grad(jnp.linalg.norm, [0])
        pass

    def test_smooth_norm_large_values(self):

        vec = np.array([10, 5, 20], dtype=float)

        test_result = math_utils.smooth_norm(vec)

        assert test_result == pytest.approx(np.linalg.norm(vec))

    def test_smooth_norm_small_values(self):

        vec = np.array([1e-6, 5e-6, 2e-6], dtype=float)

        test_result = math_utils.smooth_norm(vec)

        assert test_result == pytest.approx(np.linalg.norm(vec), abs=1e-6)

    def test_smooth_norm_zero_values(self):

        vec = np.array([0, 0, 0], dtype=float)

        test_result = math_utils.smooth_norm(vec)

        assert test_result == pytest.approx(np.linalg.norm(vec), abs=1e-6)

    def test_smooth_norm_large_values_grad(self):

        vec = np.array([10, 5, 20], dtype=float)

        test_result = self.smooth_norm_grad(vec)
        expected_result = self.norm_grad(vec)

        assert np.all(test_result[0] == pytest.approx(expected_result[0]))

    def test_smooth_norm_small_values_grad(self):

        vec = np.array([1e-6, 5e-6, 2e-6], dtype=float)

        test_result = self.smooth_norm_grad(vec)

        # loose tolerance due to expected error in the grad of smooth_norm for small vector values
        assert np.all(test_result[0] == pytest.approx(self.norm_grad(vec)[0], abs=1e-1))

    def test_smooth_norm_zero_values_grad(self):

        vec = np.array([0, 0, 0], dtype=float)

        test_result = self.smooth_norm_grad(vec)

        # zero valued gradient expected for the grad of smooth_norm for zero vector values
        assert np.all(
            test_result[0] == pytest.approx(np.array([0, 0, 0], dtype=float), abs=1e-1)
        )
