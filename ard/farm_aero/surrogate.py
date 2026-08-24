"""
Surrogate wind farm power model: a continuous N_turbines -> hourly power
lookup, built by build_surrogate.py (hybridfarm repo root, outside Ard) from
a sweep of per-N SNOPT+FLOWFarm layout optimizations.

Unlike the FarmAeroTemplate-based components in this package, this model has
no turbine x/y layout inputs -- N_turbines is a continuous scalar (so a
gradient-based driver can size the farm directly), and no particular layout
is represented: it only knows the aggregate power response vs. farm size,
fit once per N from FLOWFarm's own optimized layouts.
"""

import pickle

import numpy as np
import openmdao.api as om


class SurrogateFarmPower(om.ExplicitComponent):
    """
    Hourly farm power from the (wind_speed, wind_direction, N_turbines) ->
    power surrogate.

    Options
    -------
    surrogate_pkl_path : str
        path to outputs/surrogates/surrogate_f_power.pkl from build_surrogate.py
    wind_resource_npz_path : str
        path to outputs/sweeps/n_turbines_sweep/wind_resource.npz (hourly
        wind_speed/wind_direction arrays) from build_surrogate.py
    n_timesteps : int, optional
        number of hours (from the start of the wind resource) to evaluate --
        this is the knob for getting a short hourly-dispatch window instead
        of a full-year AEP. Defaults to the full wind-resource length.
        Requesting fewer hours is NOT rescaled to an annual estimate: it
        just returns that many hours of dispatch, in wind-resource order.

    Inputs
    ------
    N_turbines : float
        number of turbines (continuous, for gradient-based sizing)

    Outputs
    -------
    electricity_out : np.ndarray
        farm power for each of the `n_timesteps` hours, kW
    AEP : float
        sum of `electricity_out` over the requested horizon, MWh (only an
        actual annual AEP if n_timesteps covers a full year of hours)
    """

    def initialize(self):
        self.options.declare("surrogate_pkl_path", types=str)
        self.options.declare("wind_resource_npz_path", types=str)
        self.options.declare("n_timesteps", types=int, default=None)

    def setup(self):
        # build_surrogate.py pickles PerStatePowerSurrogate while running as
        # __main__, so it only unpickles cleanly if that class is also
        # visible as __main__.PerStatePowerSurrogate in the loading process.
        import sys

        import build_surrogate

        sys.modules["__main__"].PerStatePowerSurrogate = (
            build_surrogate.PerStatePowerSurrogate
        )

        with open(self.options["surrogate_pkl_path"], "rb") as fh:
            pkl = pickle.load(fh)
        self._f_interp = pkl["interpolator"]

        wr = np.load(self.options["wind_resource_npz_path"])
        n = self.options["n_timesteps"]
        self._ws = wr["wind_speed"] if n is None else wr["wind_speed"][:n]
        self._wd = wr["wind_direction"] if n is None else wr["wind_direction"][:n]
        self._n = len(self._ws)

        self.add_input("N_turbines", val=25.0)
        self.add_output("electricity_out", val=np.zeros(self._n), units="kW")
        self.add_output("AEP", val=0.0, units="MW*h")

    def setup_partials(self):
        self.declare_partials("electricity_out", "N_turbines")
        self.declare_partials("AEP", "N_turbines")

    def compute(self, inputs, outputs):
        n_turbines = inputs["N_turbines"].item()
        power_kw = np.asarray(
            self._f_interp(self._ws, self._wd, np.full(self._n, n_turbines))
        )
        # clamp small negative undershoots from the RBF/spline near cut-in
        # wind speed -- power is never negative
        power_kw = np.maximum(power_kw, 0.0)
        outputs["electricity_out"] = power_kw
        outputs["AEP"] = power_kw.sum() / 1000.0

    def compute_partials(self, inputs, partials):
        n_turbines = inputs["N_turbines"].item()
        power_kw = np.asarray(
            self._f_interp(self._ws, self._wd, np.full(self._n, n_turbines))
        )
        dpower_dn = np.asarray(
            self._f_interp(self._ws, self._wd, np.full(self._n, n_turbines), dz=1)
        )
        # zero derivative wherever the clamp is active, matching compute()
        dpower_dn = np.where(power_kw > 0.0, dpower_dn, 0.0)
        partials["electricity_out", "N_turbines"] = dpower_dn.reshape(-1, 1)
        partials["AEP", "N_turbines"] = dpower_dn.sum() / 1000.0
