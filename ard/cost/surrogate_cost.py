"""
Surrogate wind CapEx/OpEx model: quadratic fits of cost vs. N_turbines built
by build_surrogate.py (hybridfarm repo root, outside Ard) from a sweep of
per-N SNOPT+FLOWFarm layout optimizations.
"""

import pickle

import numpy as np
import openmdao.api as om


class SurrogateWindCost(om.ExplicitComponent):
    """
    Wind CapEx/OpEx from the g(N_turbines) polynomial cost surrogate.

    Options
    -------
    surrogate_pkl_path : str
        path to outputs/surrogate_g_cost.pkl from build_surrogate.py

    Inputs
    ------
    N_turbines : float
        number of turbines (continuous, for gradient-based sizing)

    Outputs
    -------
    CapEx : float
        wind CapEx, USD -- includes cabling cost (direct material cost plus
        its effect on LandBOSSE's spacing-driven BOS cost); cable length
        itself is not exposed here since it is not reliably fit standalone
        (validated 14-28% error vs. held-out FLOWFarm/optiwindnet runs),
        while total CapEx validated to ~1% error since cabling is only a
        small (~4-6%) share of it.
    OpEx : float
        wind annual OpEx, USD/yr
    """

    def initialize(self):
        self.options.declare("surrogate_pkl_path", types=str)

    def setup(self):
        with open(self.options["surrogate_pkl_path"], "rb") as fh:
            pkl = pickle.load(fh)
        self._capex_coeffs = pkl["poly_coeffs"]["wind_capex_usd"]
        self._opex_coeffs = pkl["poly_coeffs"]["wind_opex_usd_per_yr"]
        self._dcapex_coeffs = np.polyder(self._capex_coeffs)
        self._dopex_coeffs = np.polyder(self._opex_coeffs)

        self.add_input("N_turbines", val=25.0)
        self.add_output("CapEx", val=0.0, units="USD")
        self.add_output("OpEx", val=0.0, units="USD/yr")

    def setup_partials(self):
        self.declare_partials("CapEx", "N_turbines")
        self.declare_partials("OpEx", "N_turbines")

    def compute(self, inputs, outputs):
        n = inputs["N_turbines"].item()
        outputs["CapEx"] = np.polyval(self._capex_coeffs, n)
        outputs["OpEx"] = np.polyval(self._opex_coeffs, n)

    def compute_partials(self, inputs, partials):
        n = inputs["N_turbines"].item()
        partials["CapEx", "N_turbines"] = np.polyval(self._dcapex_coeffs, n)
        partials["OpEx", "N_turbines"] = np.polyval(self._dopex_coeffs, n)
