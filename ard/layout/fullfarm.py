import numpy as np
import openmdao.api as om

from shapely import length
import shapely.geometry as sg

import ard.layout.templates


class FullFarmLanduse(ard.layout.templates.LanduseTemplate):
    """
    Landuse class for full Cartesian grid farm layout.

    This is a class to compute the landuse area of a fully specified Cartesian
    grid farm layout.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from
        `templates.LanduseTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1-D numpy array that represents that x (i.e. Easting) coordinate of
        the location of each of the turbines in the farm in meters
    y_turbines : np.ndarray
        a 1-D numpy array that represents that y (i.e. Northing) coordinate of
        the location of each of the turbines in the farm in meters

    Outputs
    -------
    area_tight : float
        the area in square kilometers that the farm occupies based on the
        circumscribing geometry with a specified (default zero) layback buffer
        (inherited from `templates.LayoutTemplate`)
    """

    def setup(self):
        """Setup of OM component."""
        super().setup()

        # add the full layout inputs
        self.add_input(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in x-direction",
        )
        self.add_input(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in y-direction",
        )

    def setup_partials(self):
        """Derivative setup for OM component."""

        # default complex step for the layout-landuse tools, since they're often algebraic
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """Computation for the OM component."""

        # extract the points from the inputs
        points = list(
            zip(
                list(inputs["x_turbines"]),
                list(inputs["y_turbines"]),
            )
        )

        # create a multi-point object
        mp = sg.MultiPoint(points)

        # create a laybacked geometry
        D_rotor = self.windIO["wind_farm"]["turbine"]["rotor_diameter"]
        lengthscale_layback = float(inputs["distance_layback_diameters"][0] * D_rotor)

        # area tight is equal to the convex hull area for the points in sq. km.
        area_tight = mp.convex_hull.buffer(lengthscale_layback).area / 1000**2
        outputs["area_tight"] = mp.convex_hull.buffer(lengthscale_layback).area / 1000**2
