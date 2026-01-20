import numpy as np
from scipy.interpolate import RectBivariateSpline

import openmdao.api as om


class EagleDensityFunction(om.ExplicitComponent):
    """
    OpenMDAO component to evaluate eagle presence density at turbine locations.

    An Ard/OpenMDAO component that evaluates the eagle presence density metric
    calculated by the National Laboratory of the Rockies's Stochastic Soaring
    Raptor Simulator (SSRS) at the turbine locations. The eagle presence density
    is an output of an SSRS simulation indicating the unit density function of
    a raptor flying through the point during a given migratory period.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from
        `templates.LanduseTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1-D numpy array that represents the x (i.e. Easting) coordinate of
        the location of each of the turbines in the farm in meters
    y_turbines : np.ndarray
        a 1-D numpy array that represents the y (i.e. Northing) coordinate of
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

        self.N_turbines = modeling_options["layout"]["N_turbines"]

        # grab the eagle presence density settings
        self.pres = self.modeling_options["eco"]["eagle_presence_density_map"]
        self.eagle_density_function = RectBivariateSpline(
            self.pres["x"], self.pres["y"], self.pres["normalized_presence_density"]
        )
        self.eagle_density_function_dx = self.eagle_density_function.partial_derivative(
            dx=1, dy=0
        )
        self.eagle_density_function_dy = self.eagle_density_function.partial_derivative(
            dx=0, dy=1
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
            units=None,
            desc="normalized eagle presence density",
        )

    def setup_partials(self):
        """Setup the OpenMDAO component partial derivatives."""
        self.declare_partials(
            "eagle_normalized_density",
            "x_turbines",
            diagonal=True,
            method="exact",
        )
        self.declare_partials(
            "eagle_normalized_density",
            "y_turbines",
            diagonal=True,
            method="exact",
        )

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.
        """

        # unpack the turbine locations
        x_turbines = inputs["x_turbines"]  # m
        y_turbines = inputs["y_turbines"]  # m

        # evaluate the density function at each turbine point
        outputs["eagle_normalized_density"] = self.eagle_density_function(
            x_turbines, y_turbines, grid=False,
        )

    def compute_partials(self, inputs, partials):
        """
        Compute the partials for the OM component
        """

        # unpack the turbine locations
        x_turbines = inputs["x_turbines"]  # m
        y_turbines = inputs["y_turbines"]  # m

        # evaluate the gradients for each variable
        dfdx = self.eagle_density_function_dx(x_turbines, y_turbines, grid=False)
        dfdy = self.eagle_density_function_dy(x_turbines, y_turbines, grid=False)
        partials["eagle_normalized_density", "x_turbines"] = dfdx
        partials["eagle_normalized_density", "y_turbines"] = dfdy
