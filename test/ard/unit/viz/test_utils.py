import numpy as np

import ard.viz.utils as viz_utils


def test_get_plot_range(subtests):
    """
    test the get_plot_range function by feeding it multiple arguments for
    matches against gold standard values
    """

    values_args_truths = [
        (
            np.array([4.0, 5.0, 6.0]),
            None,
            (4.0 - (6.0 - 4.0) * 0.05, 6.0 + (6.0 - 4.0) * 0.05),
        ),
        (
            np.linspace(-3.0, 10.0, 25),
            None,
            (-3.0 - (10.0 - -3.0) * 0.05, 10.0 + (10.0 - -3.0) * 0.05),
        ),
        (
            np.linspace(-3.0, 10.0, 25),
            7.5,
            (-3.0 - (10.0 - -3.0) * 0.075, 10.0 + (10.0 - -3.0) * 0.075),
        ),
    ]

    for idx, (value, arg, truth) in enumerate(values_args_truths):
        with subtests.test(f"value {idx:02d}"):
            if arg is None:
                rv = viz_utils.get_plot_range(value)
            else:
                rv = viz_utils.get_plot_range(value, arg)
            assert np.allclose(rv, truth)
