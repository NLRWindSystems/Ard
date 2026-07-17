import numpy as np

import openmdao.api as om


class LayoutTemplate(om.ExplicitComponent):
    """
    A template class for layout parametrizations.

    This is a template class that represents the fundamental input/output flow
    for a layout parametrization. This means outputting $(x,y)$ tuples (via
    `x_turbines` and `y_turbines) for the `N_turbines` in the farm, and
    outputting effective spacing metrics (for, e.g., simple BOS tools).

    Options
    -------
    modeling_options : dict
        a modeling options dictionary

    Inputs
    ------
    None

    Outputs
    -------
    x_turbines : np.ndarray
        a 1-D numpy array that represents that x (i.e. Easting) coordinate of
        the location of each of the turbines in the farm in meters
    y_turbines : np.ndarray
        a 1-D numpy array that represents that y (i.e. Northing) coordinate of
        the location of each of the turbines in the farm in meters
    spacing_effective_primary : float
        a measure of the spacing on a primary axis of a rectangular farm that
        would be equivalent to this one for the purposes of computing BOS costs
        measured in rotor diameters
    spacing_effective_secondary : float
        a measure of the spacing on a secondary axis of a rectangular farm that
        would be equivalent to this one for the purposes of computing BOS costs
        measured in rotor diameters
    """

    def initialize(self):
        """Initialization of OM component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of OM component."""

        # load modeling options
        modeling_options = self.modeling_options = self.options["modeling_options"]
        self.windIO = self.modeling_options["windIO_plant"]
        self.N_turbines = modeling_options["layout"]["N_turbines"]

        # add outputs that are universal
        self.add_output(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in x-direction",
        )
        self.add_output(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in y-direction",
        )
        self.add_output(
            "yaw_turbines",
            np.zeros((self.N_turbines,)),
            units="deg",
            desc="turbine yaw angle from modeling options",
        )
        self.add_output(
            "spacing_effective_primary",
            0.0,
            desc="effective spacing in x-dimension for BOS calculation",
        )
        self.add_output(
            "spacing_effective_secondary",
            0.0,
            desc="effective spacing in y-dimension for BOS calculation",
        )

    # omit setup partials for template class

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement"
        )


class LanduseTemplate(om.ExplicitComponent):
    """
    A template class for landuse computations.

    This is a template class that represents the fundamental input/output flow
    for a landuse calculation. Most details will be specialized based on use
    case, but most fundamentally it will intake a layback distance and output a
    simple area computation.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary

    Inputs
    ------
    distance_layback_diameters : float
        the number of diameters of layback desired for the landuse calculation

    Outputs
    -------
    area_tight : float
        the area in square kilometers that the farm occupies based on the
        circumscribing geometry with a specified (default zero) layback buffer
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

        # add inputs that are universal
        self.add_input(
            "distance_layback_diameters",
            0.0,
            units=None,
            desc="number of diameters of layback necessary for landuse",
        )

        # add outputs that are universal
        self.add_output(
            "area_tight",
            0.0,
            units="km**2",
            desc="fundamental area of the farm geometry",
        )

    # omit setup partials for template class

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement"
        )
