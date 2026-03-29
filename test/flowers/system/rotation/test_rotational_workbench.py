import numpy as np
import pandas as pd

import flowers


# create wraparound gaussian pulse resource for nearly single-directional wind
def f_windrose(wd_center, ws_center, wd_width=10.0, ws_width=5.0):
    wd_raw = np.arange(0.0, 360.0, 5.0)[1:]
    ws_raw = np.arange(0.0, 25.001, 1.0)[1:]
    WS, WD = np.meshgrid(ws_raw, wd_raw)
    FREQ = (np.exp(-(((WS - ws_center) / ws_width) ** 2))) * (
        np.exp(-(((WD - wd_center - 720.0) / wd_width) ** 2))
        + np.exp(-(((WD - wd_center - 360.0) / wd_width) ** 2))
        + np.exp(-(((WD - wd_center + 0.0) / wd_width) ** 2))
        + np.exp(-(((WD - wd_center + 360.0) / wd_width) ** 2))
        + np.exp(-(((WD - wd_center + 720.0) / wd_width) ** 2))
    )
    FREQ = np.maximum(FREQ, 0.001)
    FREQ /= np.sum(FREQ)
    df_wr = pd.DataFrame(
        {
            "wd": WD.flat,
            "ws": WS.flat,
            "freq_val": FREQ.flat,
        }
    )
    return (WD, WS, FREQ), df_wr


# generate the layout generator w/o rotational symmetries
def layout_generator(
    diameter=1.0,
    orientation=0.0,
    factor=10 / 6,
    spacing_base=5.0,
    smash_over=True,
):
    spacing_index_x = spacing_base * np.arange(-2, 2.1)
    spacing_index_y = spacing_base * np.arange(-2, 2.1)
    max_idx = np.max(np.abs(spacing_index_y))
    x_out_list = []
    y_out_list = []
    for idx in spacing_index_y:
        here_factor = factor ** (idx / max_idx)
        x_candidate = here_factor * spacing_index_x
        if smash_over:
            x_candidate = x_candidate - np.min(x_candidate)
            x_candidate = x_candidate + np.min(spacing_index_x)
        x_out_list.append(x_candidate)
        y_out_list.append(idx * np.ones_like(spacing_index_x))

    X_pre = diameter * np.array(x_out_list).flatten()
    Y_pre = diameter * np.array(y_out_list).flatten()

    X_post, Y_post = np.array(
        [
            [np.cos(np.radians(orientation)), np.sin(np.radians(orientation))],
            [-np.sin(np.radians(orientation)), np.cos(np.radians(orientation))],
        ]
    ) @ np.vstack([X_pre, Y_pre])

    return X_post, Y_post


def run_FLOWERS(flowers_turbine, wd_val=0.0, orientation=0.0):
    X, Y = layout_generator(flowers_turbine["D"], orientation)

    fm = flowers.FlowersModel(
        wind_rose=f_windrose(
            wd_center=wd_val,
            ws_center=8.0,
        )[1],
        layout_x=X,
        layout_y=Y,
        turbine=flowers_turbine,
    )

    return fm.calculate_aep() / 1e9  # return in GWh
