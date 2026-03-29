import os
from pathlib import Path

import numpy as np
import numpy.linalg as nLA

import matplotlib.pyplot as plt

import flowers
from flowers.flowers_model import FlowersModel
import flowers.tools
import flowers.visualization

import pytest


def test_flowers_stack_nrel_5MW(subtests, show_plots=False):
    """
    a test that puts a bunch of the pieces of FLOWERS together and makes sure
    they match pyrite values
    """

    N_turb = 256
    seed_val = int.from_bytes(b"0xAR")  # lock the seed for consistent results

    boundaries = [
        (-5000.0, 0.0),
        (0.0, 5000.0),
        (5000.0, 0.0),
        (0.0, -5000.0),
    ]  # a big diamond grid

    # change the cwd
    CWD = os.getcwd()
    path_wd = Path(__file__).parent
    os.chdir(path_wd)

    # load and (optionally) show the wind rose
    df_wr = flowers.tools.load_wind_rose(4)
    flowers.visualization.plot_wind_rose(df_wr)
    plt.show() if show_plots else plt.close("all")

    # change the cwd back
    os.chdir(CWD)

    # generate and (optionally) show the layout
    discrete_layout = flowers.tools.discrete_layout(n_turb=N_turb, seed_val=seed_val)
    flowers.visualization.plot_layout(discrete_layout[0], discrete_layout[1])

    # run FLOWERS to get AEP
    flowers_model_discrete = FlowersModel(
        wind_rose=df_wr,
        layout_x=discrete_layout[0],
        layout_y=discrete_layout[1],
        turbine="nrel_5MW",
    )
    AEP_d, dAEP_d = [
        v / 1.0e9 for v in flowers_model_discrete.calculate_aep(gradient=True)
    ]
    AEP_d_ref = 5444.8491956346525  # pyrite value generated 22 Jan 2026
    norm_dAEP_d_ref = 0.109870151788895  # pyrite value generated 22 Jan 2026
    print(f"discrete farm AEP: {AEP_d} GW")
    print(f"\tderivative magnitude: {nLA.norm(dAEP_d)} GW/m")
    plt.show() if show_plots else plt.close("all")

    with subtests.test("discrete farm AEP"):
        assert np.isclose(AEP_d, AEP_d_ref)
    with subtests.test("discrete farm dAEP norm"):
        assert np.isclose(nLA.norm(dAEP_d), norm_dAEP_d_ref)

    random_layout = flowers.tools.random_layout(
        boundaries=boundaries, n_turb=N_turb, seed_val=seed_val
    )
    flowers.visualization.plot_layout(random_layout[0], random_layout[1])

    flowers_model_random = FlowersModel(
        wind_rose=df_wr,
        layout_x=random_layout[0],
        layout_y=random_layout[1],
        turbine="nrel_5MW",
    )
    AEP_r, dAEP_r = [
        v / 1.0e9 for v in flowers_model_random.calculate_aep(gradient=True)
    ]
    AEP_r_ref = 1726.9377768748438  # pyrite value generated 22 Jan 2026
    norm_dAEP_r_ref = 0.16130660604162644  # pyrite value generated 22 Jan 2026
    print(f"random farm AEP: {AEP_r}")
    print(f"\tderivative magnitude: {nLA.norm(dAEP_r)} GW/m")
    plt.show() if show_plots else plt.close("all")

    with subtests.test("random farm AEP"):
        assert np.isclose(AEP_r, AEP_r_ref)
    with subtests.test("random farm dAEP norm"):
        assert np.isclose(nLA.norm(dAEP_r), norm_dAEP_r_ref)


if __name__ == "__main__":

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def test(str):
            return DummyContext()

    test_flowers_stack_nrel_5MW(DummyContext, show_plots=True)
