import numpy as np
from scipy.interpolate import RectBivariateSpline

import openmdao.api as om


class EagleDensityFunction(om.ExplicitComponent):
    """
    OpenMDAO component to evaluate eagle presence density at turbine locations.

    An Ard/OpenMDAO component that evaluates the SRSS-generated eagle presence
    density at turbine locations, nominally for analysis of and optimization
    with respect to the likelihood of eagle interactions at the turbine
    locations.

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
    eagle_normalized_density : np.ndarray
        a 1-D numpy array that represents the normalized eagle presence density
        at each of the turbine locations (unitless)
    """

    def initialize(self):
        """Initialization of OM component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of OM component."""

        # load modeling options and turbine count
        modeling_options = self.modeling_options = self.options["modeling_options"]
        self.windIO = self.modeling_options["windIO_plant"]
        self.N_turbines = modeling_options["layout"]["N_turbines"]

        # grab the eagle presence density settings
        self.pres = self.modeling_options["ssrs"]["presence_density_map"]
        self.eagle_density_function = RectBivariateSpline(
            self.pres["x"], self.pres["y"], self.pres["normalized_presence_density"]
        )

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

        # add outputs that are universal
        self.add_output(
            "eagle_normalized_density",
            np.zeros((self.N_turbines,)),
            units="unitless",
            desc="normalized eagle presence density",
        )

    def setup_partials(self):
        """Setup the OpenMDAO component partial derivatives."""
        self.declare_partials("eagle_normalized_density", "x_turbines", method="exact")
        self.declare_partials("eagle_normalized_density", "y_turbines", method="exact")

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.
        """

        # unpack the turbine locations
        x_turbines = inputs["x_turbines"]  # m
        y_turbines = inputs["y_turbines"]  # m

        # evaluate the density function at each turbine point
        outputs["eagle_normalized_density"] = [
            self.eagle_density_function(xt, yt)
            for xt, yt in zip(x_turbines, y_turbines)
        ]

    def compute_partials(self, inputs, partials):
        """
        Compute the partials for the OM component
        """

        # unpack the turbine locations
        x_turbines = inputs["x_turbines"]  # m
        y_turbines = inputs["y_turbines"]  # m

        # evaluate the gradients for each variable
        dfdx = self.eagle_density_function.ev(x_turbines, y_turbines, dx=1, dy=0)
        dfdy = self.eagle_density_function.ev(x_turbines, y_turbines, dx=0, dy=1)
        partials["eagle_normalized_density", "x_turbines"] = np.diag(dfdx)
        partials["eagle_normalized_density", "y_turbines"] = np.diag(dfdy)
