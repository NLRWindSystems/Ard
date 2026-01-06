import numpy as np
import openmdao.api as om

import matplotlib.pyplot as plt

import pytest

import ard.layout.fullfarm as fullfarm


class TestFullFarmLanduse:

    def setup_method(self):

        self.N_turbines = 25
        self.D_rotor = 130.0

        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    },
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
            },
        }

        self.model = om.Group()
        self.lu = self.model.add_subsystem(
            "fullfarm",
            fullfarm.FullFarmLanduse(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_layout_simple(self):

        # demo layout: 7D spacing 5x5
        X, Y = [
            7.0 * 130.0 * v
            for v in np.meshgrid(np.arange(-2, 2 + 1, 1), np.arange(-2, 2 + 1, 1))
        ]
        self.prob.set_val("x_turbines", X.flatten())
        self.prob.set_val("y_turbines", Y.flatten())

        # compute reference area in sq km
        A_ref = (4.0 * 7.0 * 130.0) ** 2 / 1000**2

        # run the model
        self.prob.run_model()

        # compute the area using shapely
        area_computed = self.prob.get_val("area_tight", units="km**2")

        # assert a match
        assert np.isclose(area_computed, A_ref)

    def test_layout_circle(self):

        # demo layout: 10D radius circle
        THETA = np.linspace(0, 2 * np.pi, 25 + 1)[:-1]
        X, Y = [
            10.0 * self.D_rotor * np.sin(THETA),
            10.0 * self.D_rotor * np.cos(THETA),
        ]
        self.prob.set_val("x_turbines", X.flatten())
        self.prob.set_val("y_turbines", Y.flatten())

        # compute reference area in sq km
        A_ref = (np.pi * (10.0 * self.D_rotor) ** 2) / 1000**2

        # run the model
        self.prob.run_model()

        # compute the area using shapely
        area_computed = self.prob.get_val("area_tight", units="km**2")

        # assert a match
        assert np.isclose(area_computed, A_ref, rtol=5.0e-2)  # within 5% of circle area
