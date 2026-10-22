import numpy as np
import pytest

import ard.farm_loads.surrogate_load_functions as surrogate


@pytest.fixture
def ann_params():
    """Build a sparse network with hand-computable predictions."""
    first_weights = np.zeros((60, 10))
    first_weights[0, [0, 9]] = [2.0, 1.0]
    first_weights[1, 1] = -1.0
    first_bias = np.zeros(60)
    first_bias[:2] = [1.0, -0.5]
    second_weights = np.zeros((1, 60))
    second_weights[0, :2] = [3.0, -2.0]
    return {
        "zScore_norm": {
            "mu_x": np.arange(10, dtype=float),
            "sigma_x": np.arange(1, 11, dtype=float),
            "mu_t": 2.0,
            "sigma_t": 4.0,
        },
        "weights": {
            "W1": first_weights,
            "b1": first_bias,
            "W2": second_weights,
            "b2": 0.25,
        },
    }


@pytest.mark.parametrize("name", ["BladeRoot", "Shaft", "TowerBase", "YawBearings"])
@pytest.mark.parametrize("as_lists", [False, True])
def test_ann_forward_pass(ann_params, name, as_lists):
    """Check normalization, both linear layers, ReLU, and clipping."""
    normalized = np.zeros((3, 10))
    normalized[0, [0, 1, 9]] = [1.0, 2.0, 3.0]
    normalized[1, [0, 1]] = [-1.0, -2.0]
    samples = (
        normalized * ann_params["zScore_norm"]["sigma_x"]
        + ann_params["zScore_norm"]["mu_x"]
    )
    if as_lists:
        ann_params = {
            section: {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in values.items()
            }
            for section, values in ann_params.items()
        }
        samples = samples.tolist()

    predictor = surrogate.ANN_DEL_factory(name, {name: ann_params})
    predictions = predictor(samples)

    assert predictions.shape == (3, 1)
    np.testing.assert_allclose(predictions, [[75.0], [0.0], [15.0]])
    np.testing.assert_allclose(predictor(samples[0]), [[75.0]])


@pytest.mark.parametrize("shape", [(9,), (11,), (2, 9), (2, 11)])
def test_ann_rejects_wrong_feature_count(ann_params, shape):
    """Reject samples that do not provide ten features."""
    predictor = surrogate.ANN_DEL_factory("Shaft", {"Shaft": ann_params})
    with pytest.raises(ValueError, match=r"Input must have shape \(Q, 10\)"):
        predictor(np.zeros(shape))


def test_ann_requires_selected_model():
    """Report a missing model rather than silently using another quantity."""
    predictor = surrogate.ANN_DEL_factory("TowerBase", {})
    with pytest.raises(KeyError, match="TowerBase"):
        predictor(np.zeros(10))


def test_tansig():
    """Check the sigmoid against the equivalent hyperbolic tangent."""
    values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(surrogate.tansig(values), np.tanh(values))
    assert surrogate.tansig(0.0) == 0.0


def test_mapminmax_apply():
    """Apply distinct offsets and gains to each feature across samples."""
    settings = {
        "xoffset": np.array([2.0, -3.0]),
        "gain": np.array([0.5, 2.0]),
        "ymin": -1.0,
    }
    values = np.array([[2.0, 4.0, 6.0], [-3.0, -2.0, -1.0]])
    np.testing.assert_allclose(
        surrogate.mapminmax_apply(values, settings),
        [[-1.0, 0.0, 1.0], [-1.0, 1.0, 3.0]],
    )


def test_mapminmax_reverse():
    """Recover original outputs from scaled predictions."""
    settings = {"xoffset": 2.0, "gain": 0.5, "ymin": -1.0}
    np.testing.assert_allclose(
        surrogate.mapminmax_reverse(np.array([-1.0, 0.0, 1.0]), settings),
        [2.0, 4.0, 6.0],
    )
