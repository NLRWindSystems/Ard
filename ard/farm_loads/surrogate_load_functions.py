"""Surrogate turbine-load predictions from trained artificial neural networks.

The module stores the normalization values and weights for blade-root, shaft,
tower-base, and yaw-bearing damage-equivalent-load (DEL) models. Each predictor
accepts ten features per sample, in this order: four sector-average wind speeds,
four sector-average turbulence intensities, yaw angle, and power demand.
"""

import numpy as np
import warnings


def tansig(x):
    """Apply the hyperbolic tangent sigmoid transfer function.

    Parameters
    ----------
    x : array_like
        Input value or array of values.

    Returns
    -------
    numpy.ndarray or scalar
        Values transformed to the range [-1, 1].
    """
    return 2 / (1 + np.exp(-2 * x)) - 1


def mapminmax_apply(x, settings):
    """Scale feature-by-sample data using mapminmax settings.

    Parameters
    ----------
    x : array_like
        Data with shape (N_features, N_samples).
    settings : dict
        Scaling values containing ``xoffset``, ``gain``, and ``ymin``.

    Returns
    -------
    numpy.ndarray
        Scaled data with the same shape as ``x``.
    """
    return (x - settings["xoffset"][:, np.newaxis]) * settings["gain"][
        :, np.newaxis
    ] + settings["ymin"]


def mapminmax_reverse(y, settings):
    """Reverse scaling applied with :func:`mapminmax_apply`.

    Parameters
    ----------
    y : array_like
        Scaled data.
    settings : dict
        Scaling values containing ``xoffset``, ``gain``, and ``ymin``.

    Returns
    -------
    numpy.ndarray
        Data in the original scale.
    """
    return (y - settings["ymin"]) / settings["gain"] + settings["xoffset"]


def ANN_DEL_factory(name_QoI, ANN_DEL_params):
    """Create a predictor for one turbine damage-equivalent-load quantity.

    Parameters
    ----------
    name_QoI : str
        Model key in ``ANN_DEL_params``: ``"BladeRoot"``, ``"Shaft"``,
        ``"TowerBase"``, or ``"YawBearings"``.
    ANN_DEL_params : dict
        Mapping from model keys to model parameters. The selected model entry
        must contain ``zScore_norm`` and ``weights`` mappings. ``zScore_norm``
        must contain ``mu_x`` and ``sigma_x`` (each with shape (10,)), and
        scalar ``mu_t`` and ``sigma_t``. ``weights`` must contain ``W1`` with
        shape (60, 10), ``b1`` with shape (60,), ``W2`` with shape (1, 60),
        and scalar ``b2``. Values may be Python sequences or NumPy arrays. For
        example::

            {
                "BladeRoot": {
                    "zScore_norm": {
                        "mu_x": [...],
                        "sigma_x": [...],
                        "mu_t": 0.0,
                        "sigma_t": 1.0,
                    },
                    "weights": {
                        "W1": [...],
                        "b1": [...],
                        "W2": [...],
                        "b2": 0.0,
                    },
                },
            }

    Returns
    -------
    callable
        A function that predicts the selected DEL for one or more samples.
        Its input has ten features per sample; its output has shape (Q, 1).
    """

    warnings.warn(
        f"""Generating surrogate load function `{name_QoI}` using `ANN_DEL_factory`.
        This is an experimental feature that requires a fixed model architecture.
        Ensure its behavior is acceptable for your use case."""
    )

    def out_fun(x1):
        """Predict the selected DEL from ten features per sample.

        Parameters
        ----------
        x1 : array_like
            One sample with shape (10,) or multiple samples with shape
            (Q, 10), ordered as described in the module documentation.

        Returns
        -------
        numpy.ndarray
            Nonnegative DEL predictions with shape (Q, 1), in the units used
            to train the selected model.

        Raises
        ------
        ValueError
            If the input does not contain exactly ten features per sample.
        KeyError
            If the selected model parameters are missing from
            ``ANN_DEL_params``.
        """
        x1 = np.array(x1, dtype=np.float64)
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x1.shape[1] != 10:
            raise ValueError("Input must have shape (Q, 10)")

        # Z-score normalization parameters (fit on training data)
        mu_x = np.array(ANN_DEL_params[name_QoI]["zScore_norm"]["mu_x"])
        sigma_x = np.array(ANN_DEL_params[name_QoI]["zScore_norm"]["sigma_x"])
        mu_t = ANN_DEL_params[name_QoI]["zScore_norm"]["mu_t"]
        sigma_t = ANN_DEL_params[name_QoI]["zScore_norm"]["sigma_t"]

        # Network weights: Linear(10->60) + ReLU + Linear(60->1)
        W1 = np.array(ANN_DEL_params[name_QoI]["weights"]["W1"])  # shape (60, 10)
        b1 = np.array(ANN_DEL_params[name_QoI]["weights"]["b1"])  # shape (60,)
        W2 = np.array(ANN_DEL_params[name_QoI]["weights"]["W2"])  # shape (1, 60)
        b2 = np.array(ANN_DEL_params[name_QoI]["weights"]["b2"])  # shape (1,)

        # Forward pass: z-score normalize -> Linear + ReLU -> Linear -> z-score denormalize
        x_norm = (x1 - mu_x) / sigma_x  # (Q, 10)
        a1 = np.maximum(0, x_norm @ W1.T + b1)  # (Q, 60)  ReLU
        y_norm = a1 @ W2.T + b2  # (Q, 1)
        y = y_norm * sigma_t + mu_t  # (Q, 1)

        # remove negative entries
        y = np.maximum(0.0, y)

        return y

    return out_fun
