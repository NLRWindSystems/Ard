import numpy as np


def get_plot_range(values, pct_buffer=5.0):
    min_value = np.min(values)
    max_value = np.max(values)
    dvalues = max_value - min_value
    min_value = min_value - pct_buffer / 100.0 * dvalues
    max_value = max_value + pct_buffer / 100.0 * dvalues
    return min_value, max_value
