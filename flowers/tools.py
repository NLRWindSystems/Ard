import copy

import numpy as np
import pandas as pd
import pickle
from shapely.geometry import Polygon, Point

# import floris.tools.wind_rose as rose


###########################################################################
# Layout generation
###########################################################################


def random_layout(boundaries=[], n_turb=0, D=126.0, min_dist=2.0, seed_val=None):
    """
    Generate a random wind farm layout within the specified boundaries.
    Minimum spacing between turbines is 2D.

    Args:
        boundaries (list(tuple)): boundary vertices in the form
            [(x0,y0), (x1,y1), ... , (xN,yN)]
        n_turb (int): number of turbines
        D (float): rotor diameter [m]
        min_dist (float): enforced minimum spacing between turbine centers
            normalized by rotor diameter
        seed_val (int, optional): random number generator seed

    Args:
        xx (np.array): x-positions of each turbine
        yy (np.array): y-positions of each turbine

    """

    print("Generating wind farm layout.")

    # Verify that boundaries and turbines are supplied
    if not boundaries:
        raise ValueError("Must supply boundaries to generate wind farm.")

    if n_turb <= 0:
        raise ValueError("Must supply number of turbines.")

    # Initialize RNG and containers
    if seed_val != None:
        np.random.seed(seed_val)

    xx = -1000 * np.ones(n_turb)
    yy = -1000 * np.ones(n_turb)

    xmin = np.min([tup[0] for tup in boundaries])
    xmax = np.max([tup[0] for tup in boundaries])
    ymin = np.min([tup[1] for tup in boundaries])
    ymax = np.max([tup[1] for tup in boundaries])

    # Generate boundary polygon
    poly = Polygon(boundaries)

    # Generate new positions and check minimum spacing and boundary
    for i in range(n_turb):
        prop_x = np.random.uniform(low=xmin, high=xmax)
        prop_y = np.random.uniform(low=ymin, high=ymax)
        pt = Point(prop_x, prop_y)
        while np.any(
            np.sqrt((prop_x - xx) ** 2 + (prop_y - yy) ** 2) < min_dist * D
        ) or not pt.within(poly):
            prop_x = np.random.uniform(low=xmin, high=xmax)
            prop_y = np.random.uniform(low=ymin, high=ymax)
            pt = Point(prop_x, prop_y)
        xx[i] = prop_x
        yy[i] = prop_y

    return xx, yy


def discrete_layout(n_turb=0, D=126.0, min_dist=3.0, seed_val=None, spacing=False):
    """
    Generate a random wind farm layout within the specified boundaries.
    Minimum spacing between turbines is 2D.

    Args:
        boundaries (list(tuple)): boundary vertices in the form
            [(x0,y0), (x1,y1), ... , (xN,yN)]
        n_turb (int): number of turbines
        D (float): rotor diameter [m]
        min_dist (float): enforced minimum spacing between turbine centers
            normalized by rotor diameter
        idx (int, optional): random number generator seed

    Args:
        xx (np.array): x-positions of each turbine
        yy (np.array): y-positions of each turbine

    """

    print("Generating wind farm layout.")

    if n_turb <= 0:
        raise ValueError("Must supply number of turbines.")

    # Initialize RNG and containers
    if seed_val != None:
        np.random.seed(seed_val)

    # Indices of discrete grid
    sx = n_turb + 1
    sy = 6
    x_idx = np.random.randint(0, sx, n_turb)
    y_idx = np.random.randint(0, sy, n_turb)
    pts = [(x_idx[i], y_idx[i]) for i in range(n_turb)]
    while len(np.unique(pts)) < len(pts):
        tmp = np.unique(pts)
        new_set = []
        for i in range(n_turb):
            if i not in tmp:
                new_set.append(i)
        x_idx[new_set] = np.random.randint(0, sx, len(new_set))
        y_idx[new_set] = np.random.randint(0, sy, len(new_set))
        pts = [(x_idx[i], y_idx[i]) for i in range(n_turb)]

    # Check that all combinations of x,y are unique

    xx = np.array(min_dist * D * x_idx)
    yy = np.array(min_dist * D * y_idx)

    if spacing:
        x_rel = (xx - np.reshape(xx, (-1, 1))) / D
        y_rel = (yy - np.reshape(yy, (-1, 1))) / D
        r_rel = np.sqrt(x_rel**2 + y_rel**2)
        r_rel = np.ma.masked_where(np.eye(len(xx)), r_rel)
        ss = np.mean(np.min(r_rel, -1))
        return xx, yy, ss
    else:
        return xx, yy


def load_layout(idx, case, boundaries=True):
    file = "./layouts/" + case + str(idx) + ".p"
    with open(file, "rb") as infile:
        layout_x, layout_y, boundaries_val = pickle.load(infile)

    if boundaries:
        return layout_x, layout_y, boundaries_val
    else:
        return layout_x, layout_y


###########################################################################
# Wind rose sampling
###########################################################################


def load_wind_rose(idx):
    """
    Load a locally-stored wind rose saved to a pickle file.
    See show_wind_roses.py to visualize all wind rose options.

    Args:
        idx (int): index of desired wind rose

    Returns:
        df (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction

    """

    print("Generating wind rose.")
    file_name = "./wind_roses/wr" + str(idx) + ".p"
    df = pd.read_pickle(file_name)

    return df


def resample_wind_direction(df, wd=np.arange(0, 360, 5.0)):
    """
    Resample wind direction bins using new specified bin center values.
    (Copied from FLORIS)

    Args:
        df (pandas.DataFrame): Wind rose DataFrame containing the following
            columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Wind speed bin center values (m/s).
            - 'freq_val': The frequency of occurence of the
                wind conditions in the other columns.

        wd (np.array, optional): List of new wind direction center bins
            (deg). Defaults to np.arange(0, 360, 5.).

    Returns:
        New wind rose DataFrame containing the following columns:
            - 'wd': New wind direction bin center values from wd argument (deg).
            - 'ws': Resampled wind speed bin center values (m/s).
            - 'freq_val': The resampled frequency of occurence of the
                wind conditions in the other columns.
    """

    # Make a copy of incoming dataframe
    df = df.copy(deep=True)

    # Get the wind step
    wd_step = wd[1] - wd[0]

    # Get bin edges
    wd_edges = wd - wd_step / 2.0
    wd_edges = np.append(wd_edges, np.array(wd[-1] + wd_step / 2.0))

    # Get the overhangs
    negative_overhang = wd_edges[0]
    positive_overhang = wd_edges[-1] - 360.0

    # Potentially wrap high angle direction to negative for correct binning
    tmp = df.wd
    tmp = np.where(tmp < 0.0, tmp + 360.0, tmp)
    tmp = np.where(tmp >= 360.0, tmp - 360.0, tmp)
    df["wd"] = tmp

    if negative_overhang < 0:
        # print("Correcting negative Overhang:%.1f" % negative_overhang)
        df["wd"] = np.where(
            df.wd.values >= 360.0 + negative_overhang,
            df.wd.values - 360.0,
            df.wd.values,
        )

    # Check on other side
    if positive_overhang > 0:
        # print("Correcting positive Overhang:%.1f" % positive_overhang)
        df["wd"] = np.where(
            df.wd.values <= positive_overhang, df.wd.values + 360.0, df.wd.values
        )

    # Cut into bins
    df["wd"] = pd.cut(df.wd, wd_edges, labels=wd)

    # Regroup
    df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()

    # Fill nans
    df = df.fillna(0)

    # Reset the index
    df = df.reset_index()

    # Set to float Re-wrap
    for c in [c for c in df.columns if c != "freq_val"]:
        df[c] = df[c].astype(float)
        df[c] = df[c].astype(float)

    tmp = df.wd
    tmp = np.where(tmp < 0.0, tmp + 360.0, tmp)
    tmp = np.where(tmp >= 360.0, tmp - 360.0, tmp)
    df["wd"] = tmp

    return df


def resample_wind_speed(df, ws=np.arange(0, 26, 1.0)):
    """
    Resample wind speed bins using new specified bin center values.
    (Copied from FLORIS)

    Args:
        df (pandas.DataFrame): Wind rose DataFrame containing the following
            columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Wind speed bin center values (m/s).
            - 'freq_val': The frequency of occurence of the
                wind conditions in the other columns.

        ws (np.array, optional): List of new wind direction center bins
            (m/s). Defaults to np.arange(0, 26, 1.0).

    Returns:
        New wind rose DataFrame containing the following columns:
            - 'wd': New wind direction bin center values from wd argument (deg).
            - 'ws': Resampled wind speed bin center values (m/s).
            - 'freq_val': The resampled frequency of occurence of the
                wind conditions in the other columns.
    """
    # Make a copy of incoming dataframe
    df = df.copy(deep=True)

    # Get the wind step
    ws_step = ws[1] - ws[0]

    # Ws
    ws_edges = ws - ws_step / 2.0
    ws_edges = np.append(ws_edges, np.array(ws[-1] + ws_step / 2.0))

    # Cut wind speed onto bins
    df["ws"] = pd.cut(df.ws, ws_edges, labels=ws)

    # Regroup
    df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()

    # Fill nans
    df = df.fillna(0)

    # Reset the index
    df = df.reset_index()

    # Set to float
    for c in [c for c in df.columns if c != "freq_val"]:
        df[c] = df[c].astype(float)
        df[c] = df[c].astype(float)

    return df


def resample_average_ws_by_wd(wind_rose):
    """
        Calculate the mean wind speed for each wind direction bin
        and resample the wind rose. (Copied from FLORIS)

    Args:
        wind_rose (WindRose): Wind rose object containing the following
            columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Wind speed bin center values (m/s).
            - 'freq_val': The frequency of occurence of the
                wind conditions in the other columns.

    Returns:
        New wind rose DataFrame containing the following columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Resampled average wind speed values (m/s).
            - 'freq_val': The resampled frequency of occurence of the
                wind conditions in the other columns.

    """
    # Make a copy of incoming FLORIS WindRose object
    df = copy.deepcopy(wind_rose)
    # wind_rose = copy.deepcopy(wind_rose)

    ws_avg = []

    for val in df.wd.unique():
        ws_avg.append(
            np.array(
                df.loc[df["wd"] == val]["ws"] * df.loc[df["wd"] == val]["freq_val"]
            ).sum()
            / df.loc[df["wd"] == val]["freq_val"].sum()
        )

    # freq_avg = np.sum(wind_rose.freq_table, axis=1)
    # ws_avg = np.sum(
    #     wind_rose.freq_table * wind_rose.wind_speeds.reshape(1, -1), axis=1
    # ) / freq_avg

    # print(wind_rose.wind_directions)
    # print(wind_rose.freq_table)
    # print(ws_avg)
    # lkj
    # df = pd.DataFrame()
    # df["wd"] = wind_rose.wind_directions
    # df["ws"] = ws_avg
    # df["freq_val"] = freq_avg

    # Regroup
    df = df.groupby("wd").sum()

    df["ws"] = ws_avg

    # Reset the index
    df = df.reset_index()

    # Set to float
    df["ws"] = df.ws.astype(float)
    df["wd"] = df.wd.astype(float)

    return df


###########################################################################
# Turbine parameter tables
###########################################################################


def ct_lookup(u, turbine_type, ct=None):
    """
    Look-up table for thrust coefficient of the NREL 5 MW turbine.

    Args:
        u (float): normalized inflow wind speed

    Returns:
        ct (float): thrust coefficient

    """

    if ct != None:
        ct_table = np.array([0.0, 0.0, ct, ct, 0.0, 0.0])
        u_table = 1 / 25.0 * np.array([0.0, 2.0, 2.5, 25.01, 25.02, 50.0])
    elif turbine_type == "nrel_5MW":
        ct_table = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.99,
                0.99,
                0.97373036,
                0.92826162,
                0.89210543,
                0.86100905,
                0.835423,
                0.81237673,
                0.79225789,
                0.77584769,
                0.7629228,
                0.76156073,
                0.76261984,
                0.76169723,
                0.75232027,
                0.74026851,
                0.72987175,
                0.70701647,
                0.54054532,
                0.45509459,
                0.39343381,
                0.34250785,
                0.30487242,
                0.27164979,
                0.24361964,
                0.21973831,
                0.19918151,
                0.18131868,
                0.16537679,
                0.15103727,
                0.13998636,
                0.1289037,
                0.11970413,
                0.11087113,
                0.10339901,
                0.09617888,
                0.09009926,
                0.08395078,
                0.0791188,
                0.07448356,
                0.07050731,
                0.06684119,
                0.06345518,
                0.06032267,
                0.05741999,
                0.05472609,
                0.0,
                0.0,
            ]
        )
        u_table = (
            1
            / 25.0
            * np.array(
                [
                    0.0,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                    8.5,
                    9.0,
                    9.5,
                    10.0,
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    15.5,
                    16.0,
                    16.5,
                    17.0,
                    17.5,
                    18.0,
                    18.5,
                    19.0,
                    19.5,
                    20.0,
                    20.5,
                    21.0,
                    21.5,
                    22.0,
                    22.5,
                    23.0,
                    23.5,
                    24.0,
                    24.5,
                    25.0,
                    25.01,
                    25.02,
                    50.0,
                ]
            )
        )
    else:
        raise NotImplementedError("your specified turbine type was not found.")

    return np.interp(u, u_table, ct_table)


def cp_lookup(u, turbine_type, cp=None):
    """
    Look-up table for power coefficient of the NREL 5 MW turbine.

    Args:
        u (float): normalized inflow wind speed

    Returns:
        cp (float): power coefficient

    """
    if cp != None:
        cp_table = np.array([0.0, 0.0, cp, cp, 0.0, 0.0])
        u_table = 1 / 25.0 * np.array([0.0, 2.0, 2.5, 25.01, 25.02, 50.0])
    elif turbine_type == "nrel_5MW":
        cp_table = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.178085,
                0.289075,
                0.349022,
                0.384728,
                0.406059,
                0.420228,
                0.428823,
                0.433873,
                0.436223,
                0.436845,
                0.436575,
                0.436511,
                0.436561,
                0.436517,
                0.435903,
                0.434673,
                0.433230,
                0.430466,
                0.378869,
                0.335199,
                0.297991,
                0.266092,
                0.238588,
                0.214748,
                0.193981,
                0.175808,
                0.159835,
                0.145741,
                0.133256,
                0.122157,
                0.112257,
                0.103399,
                0.095449,
                0.088294,
                0.081836,
                0.075993,
                0.070692,
                0.065875,
                0.061484,
                0.057476,
                0.053809,
                0.050447,
                0.047358,
                0.044518,
                0.041900,
                0.039483,
                0.0,
                0.0,
            ]
        )
        u_table = (
            1
            / 25.0
            * np.array(
                [
                    0.0,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                    8.5,
                    9.0,
                    9.5,
                    10.0,
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    15.5,
                    16.0,
                    16.5,
                    17.0,
                    17.5,
                    18.0,
                    18.5,
                    19.0,
                    19.5,
                    20.0,
                    20.5,
                    21.0,
                    21.5,
                    22.0,
                    22.5,
                    23.0,
                    23.5,
                    24.0,
                    24.5,
                    25.0,
                    25.01,
                    25.02,
                    50.0,
                ]
            )
        )
    else:
        raise NotImplementedError("your specified turbine type was not found.")

    return np.interp(u, u_table, cp_table)
