import numpy as np


def get_plot_range(values, pct_buffer=5.0):
    """
    get the min and max values for a plot axis with a buffer applied

    Parameters
    ----------
    values : np.array
        the array of values in a given dimension
    pct_buffer : float, optional
        percent that should be included as a buffer, by default 5.0

    Returns
    -------
    float
        minimum value for the plot range
    float
        maximum value for the plot range
    """
    min_value = np.min(values)
    max_value = np.max(values)
    dvalues = max_value - min_value
    min_value = min_value - pct_buffer / 100.0 * dvalues
    max_value = max_value + pct_buffer / 100.0 * dvalues
    return min_value, max_value
