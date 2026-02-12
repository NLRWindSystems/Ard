import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from ard.utils.test_utils import pyrite_validator
import flowers.tools

import pytest


@pytest.mark.usefixtures("subtests")
def test_random_layout(subtests):

    boundaries_values = [
        (1000.0, 1000.0),
        (-1000.0, 1000.0),
        (-1000.0, -1000.0),
        (1000.0, -1000.0),
    ]  # a diamond boundary
    n_turb = 5
    D = 127.0  # m
    min_dist = 2.5  # D_rotor
    seed_val = int.from_bytes(b"0x1e")  # create a big fixed integer for a random seed

    # make sure the non-specified boundary raises an error
    with subtests.test("no boundary error"):
        with pytest.raises(ValueError):
            flowers.tools.random_layout([])

    # make sure the non-specified turbines raises an error
    with subtests.test("no turbines error"):
        with pytest.raises(ValueError):
            flowers.tools.random_layout(boundaries_values, n_turb=0)

    # make sure you get a boundary-conforming result w/ defaults on and no seed
    xx_defaults_noseeds, yy_defaults_noseeds = flowers.tools.random_layout(
        boundaries=boundaries_values, n_turb=n_turb
    )
    with subtests.test("random_layout default noseed boundary conforming"):
        boundary_conforming = np.all(
            (xx_defaults_noseeds >= -1000.0)
            & (xx_defaults_noseeds <= 1000.0)
            & (yy_defaults_noseeds >= -1000.0)
            & (yy_defaults_noseeds < 1000.0)
        )
        assert boundary_conforming

    # make sure you get a good result w/ custom spec and no seed
    xx_spec_noseeds, yy_spec_noseeds = flowers.tools.random_layout(
        boundaries=boundaries_values, n_turb=n_turb, D=D, min_dist=min_dist
    )
    # both boundary conforming...
    with subtests.test("random_layout specified noseed boundary conforming"):
        boundary_conforming = np.all(
            (xx_spec_noseeds >= -1000.0)
            & (xx_spec_noseeds <= 1000.0)
            & (yy_spec_noseeds >= -1000.0)
            & (yy_spec_noseeds < 1000.0)
        )
        assert boundary_conforming
    # ... and constraint satisfying
    with subtests.test("random_layout specified noseed constraint satisfying"):
        dist_mtx = np.sqrt(
            (xx_spec_noseeds - np.atleast_2d(xx_spec_noseeds).T) ** 2
            + (yy_spec_noseeds - np.atleast_2d(yy_spec_noseeds).T) ** 2
        )
        np.fill_diagonal(dist_mtx, np.inf)
        assert np.all(dist_mtx >= 2 * D)

    with subtests.test("random_layout specified seed pyrite data match"):
        xx_spec_seed, yy_spec_seed = flowers.tools.random_layout(
            boundaries=boundaries_values,
            n_turb=n_turb,
            D=D,
            min_dist=min_dist,
            seed_val=seed_val,
        )
        dump_pickle = False
        if dump_pickle:
            pickle.dump(
                (
                    np.array(xx_spec_seed),
                    np.array(yy_spec_seed),
                    np.array(boundaries_values),
                ),
                open(Path(__file__).parent / "layouts" / "garbage5.p", "wb"),
            )
        pyrite_validator(
            data_for_validation={
                "xx": xx_spec_seed,
                "yy": yy_spec_seed,
                "boundaries": boundaries_values,
            },
            filename_pyrite=Path(__file__).parent / "test_tool_random_pyrite",
            rewrite=False,
        )


@pytest.mark.usefixtures("subtests")
def test_discrete_layout(subtests):

    n_turb = 5
    D = 127.0  # m
    min_dist = 2.5  # D_rotor
    seed_val = int.from_bytes(b"0xF5")  # create a big fixed integer for a random seed

    # make sure the non-specified turbines raises an error
    with subtests.test("no turbines error"):
        with pytest.raises(ValueError):
            flowers.tools.discrete_layout(n_turb=0)

    with subtests.test("discrete_layout specified seed pyrite data match"):
        xx_spec_seed, yy_spec_seed = flowers.tools.discrete_layout(
            n_turb=n_turb,
            D=D,
            min_dist=min_dist,
            seed_val=seed_val,
        )
        pyrite_validator(
            data_for_validation={"xx": xx_spec_seed, "yy": yy_spec_seed},
            filename_pyrite=Path(__file__).parent / "test_tool_discrete_pyrite",
            rewrite=False,
        )


def test_load_layout(subtests):

    # specify some figures
    idx = 5
    case = "garbage"

    # change the cwd
    CWD = os.getcwd()
    path_wd = Path(__file__).parent
    os.chdir(path_wd)

    # load the layout
    layout_x, layout_y, boundaries = flowers.tools.load_layout(
        idx, case, boundaries=True
    )

    # change the cwd back
    os.chdir(CWD)

    # validate against pyrite data
    pyrite_validator(
        data_for_validation={"xx": layout_x, "yy": layout_y, "boundaries": boundaries},
        filename_pyrite=Path(__file__).parent / "test_tool_random_pyrite",
        rewrite=False,
    )


class TestToolsWindRose:

    def setup_method(self):
        self.path_csv = (
            Path(__file__).parents[3]
            / "examples"
            / "flowers"
            / "data"
            / "HKW_wind_rose.csv"
        )
        self.df_wr = pd.read_csv(self.path_csv)
        self.len_df_wr = len(self.df_wr)

        self.wr_idx = 4

    def test_load_wind_rose(self, subtests):

        # change the cwd
        CWD = os.getcwd()
        path_wd = Path(__file__).parent
        os.chdir(path_wd)

        # use the wind rose loader
        df_wr_loaded = flowers.tools.load_wind_rose(self.wr_idx)

        # change the cwd back
        os.chdir(CWD)

        # make sure allclose on each column
        for column in self.df_wr.columns:
            with subtests.test(f"loaded df matches on {column}"):
                assert np.allclose(df_wr_loaded[column], self.df_wr[column])

    def test_resample_wind_direction(self, subtests):

        # resample the wind direction
        # wd_unique_orig = self.df_wr.wd.unique()
        df_resampled = flowers.tools.resample_wind_direction(
            self.df_wr, wd=np.arange(0, 360, 10.0)
        )
        wd_unique_new = df_resampled.wd.unique()

        # get and sort the values for comparison
        df_check = self.df_wr[self.df_wr.wd.isin(wd_unique_new)].sort_values(
            ["ws", "wd"]
        )
        df_resampled = df_resampled.sort_values(["ws", "wd"])

        # make sure allclose on each column that's not frequency
        for column in df_check:
            if column == "freq_val":
                continue  # frequency is re-computed
            with subtests.test(f"resampled df matches on {column}"):
                assert np.allclose(df_resampled[column], df_check[column])

        # resample should be approximately one
        with subtests.test(f"ensure freq_val sums to one"):
            assert np.isclose(np.sum(df_resampled.freq_val), 1.0, atol=0.025)

    def test_resample_wind_speed(self, subtests):

        # resample the wind speed
        # wd_unique_orig = self.df_wr.wd.unique()
        df_resampled = flowers.tools.resample_wind_speed(
            self.df_wr, ws=np.arange(0, 25.0, 2.0)
        )
        ws_unique_new = df_resampled.ws.unique()

        # get and sort the values for comparison
        df_check = self.df_wr[self.df_wr.ws.isin(ws_unique_new)].sort_values(
            ["ws", "wd"]
        )
        df_resampled = df_resampled.sort_values(["ws", "wd"])

        # make sure allclose on each column that's not frequency
        for column in df_check:
            if column == "freq_val":
                continue  # frequency is re-computed
            with subtests.test(f"resampled df matches on {column}"):
                assert np.allclose(df_resampled[column], df_check[column])

        # resample should be approximately one
        with subtests.test(f"ensure freq_val sums to one"):
            assert np.isclose(np.sum(df_resampled.freq_val), 1.0, atol=0.025)

    def test_resample_average_ws_by_wd(self, subtests):

        # get the resample
        df_resampled = flowers.tools.resample_average_ws_by_wd(self.df_wr)

        # check wd values
        with subtests.test(f"resampling has same wd values"):
            assert np.allclose(df_resampled.wd.unique(), self.df_wr.wd.unique())

        # check freq_val
        with subtests.test("frequency bins correct"):
            assert np.allclose(
                df_resampled.freq_val,
                self.df_wr.groupby("wd").agg({"freq_val": np.sum}).freq_val,
            )

        # check average ws
        with subtests.test("average speed correct"):
            weighted_ws = self.df_wr.groupby("wd").apply(
                lambda group: np.sum(group["ws"] * group["freq_val"])
                / np.sum(group["freq_val"])
            )
            assert np.allclose(df_resampled.ws, weighted_ws)


class TestToolsLookup:

    def setup_method(self):
        self.u_samples = [2.0, 5.0, 8.0, 12.0, 20.0, 28.0]  # check every region
        self.sample_code = {
            "regI": 0,
            "regII_lo": 1,
            "regII_hi": 2,
            "regIII_lo": 3,
            "regIII_hi": 4,
            "regIV": 5,
        }

    def test_notimplemented_raised(self, subtests):
        with subtests.test("lookup cp invalid turbine raises error"):
            with pytest.raises(NotImplementedError):
                flowers.tools.cp_lookup(
                    self.u_samples,
                    turbine_type="the_greatest_turbine",
                    # cp=None,  # defaulted to None
                )
        with subtests.test("lookup ct invalid turbine raises error"):
            with pytest.raises(NotImplementedError):
                flowers.tools.ct_lookup(
                    self.u_samples,
                    turbine_type="the_greatest_turbine",
                    # ct=None,  # defaulted to None
                )

    def test_NREL5MW(self, subtests):

        cpmax_NREL5MW = 0.436845
        ctmax_NREL5MW = 0.99

        cp_samples = flowers.tools.cp_lookup(self.u_samples, turbine_type="nrel_5MW")
        ct_samples = flowers.tools.ct_lookup(self.u_samples, turbine_type="nrel_5MW")

        with subtests.test("approx reference max cp"):
            np.isclose(np.max(cp_samples), cpmax_NREL5MW)
        with subtests.test("approx reference max ct"):
            np.isclose(np.max(ct_samples), ctmax_NREL5MW)

        with subtests.test("region I cp approx zero"):
            np.isclose(cp_samples[self.sample_code["regI"]], 0.0)
        with subtests.test("region I ct approx zero"):
            np.isclose(ct_samples[self.sample_code["regI"]], 0.0)

        with subtests.test("region IV cp approx zero"):
            np.isclose(cp_samples[self.sample_code["regIV"]], 0.0)
        with subtests.test("region IV ct approx zero"):
            np.isclose(ct_samples[self.sample_code["regIV"]], 0.0)
