import numpy as np

import ard.layout.templates as templates


class GridFarmLayout(templates.LayoutTemplate):
    """
    A simplified, uniform four-parameter parallelepiped grid farm layout class.

    This is a class to take a parameterized, structured grid farm defined by a
    gridded parallelepiped with spacing variables defined to:
    1) orient the farm with respect to North,
    2) space the rows of turbines along this primary vector,
    3) space the columns of turbines along the perpendicular, and
    4) skew the positioning along a parallel to the primary (orientation) vector.

    The layout model is shown in a ASCII image below:

    ::

    |                                          |-------| <- streamwise spacing
    |          orient.         x ----- x ----- x ----- x ----- x -
    |           angle         /       /       /       /       /  | <- spanwise spacing
    |            |           x ----- x ----- x ----- x ----- x   -      (perpendicular
    |            v          /       /       /       /       /            w.r.t. primary)
    |           -------    x ----- x ----- x ----- x ----- x    ----- primary vector
    |                '    /       /       /       /       /             (rotated from
    |            '       x ----- x ----- x ----- x ----- x               north CW by
    |        '          /       /       /       /       /                orientation
    |        NORTH     x ----- x ----- x ----- x ----- x                 angle)
    |                        /|
    |                       / |
    |                      /  | <- skew angle


    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from
        `templates.LayoutTemplate`)
    N_turbines : int
        the number of turbines that should be in the farm layout (inherited from
        `templates.LayoutTemplate`)

    Inputs
    ------
    angle_orientation : float
        orientation in degrees clockwise with respect to North of the primary
        axis of the wind farm layout
    spacing_primary : float
        spacing of turbine rows along the primary axis (rotated by
        `angle_orientation`) in nondimensional rotor diameters
    spacing_secondary : float
        spacing of turbine columns along the perpendicular to the primary axis
        (rotated by 90° with respect to the primary axis) in nondimensional
        rotor diameters
    angle_skew : float
        clockwise skew angle of turbine rows w.r.t. beyond the 90° clockwise
        perpendicular to the primary axis

    Outputs
    -------
    x_turbines : np.ndarray
        a 1-D numpy array that represents that x (i.e. Easting) coordinate of
        the location of each of the turbines in the farm in meters (inherited
        from `templates.LayoutTemplate`)
    y_turbines : np.ndarray
        a 1-D numpy array that represents that y (i.e. Northing) coordinate of
        the location of each of the turbines in the farm in meters (inherited
        from `templates.LayoutTemplate`)
    spacing_effective_primary : float
        a measure of the spacing on a primary axis of a rectangular farm that
        would be equivalent to this one for the purposes of computing BOS costs
        measured in rotor diameters (inherited from `templates.LayoutTemplate`)
    spacing_effective_secondary : float
        a measure of the spacing on a secondary axis of a rectangular farm that
        would be equivalent to this one for the purposes of computing BOS costs
        measured in rotor diameters (inherited from `templates.LayoutTemplate`)
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

    def setup(self):
        """Setup of OM component."""
        super().setup()
        x0 = self.modeling_options["layout"].get("x0", 0.0)
        y0 = self.modeling_options["layout"].get("y0", 0.0)
        spacing_primary = self.modeling_options["layout"]["spacing_primary"]
        spacing_secondary = self.modeling_options["layout"]["spacing_secondary"]
        angle_orientation = self.modeling_options["layout"]["angle_orientation"]
        angle_skew = self.modeling_options["layout"]["angle_skew"]

        # add four-parameter grid farm layout DVs
        self.add_input("x0", x0, units="m")
        self.add_input("y0", y0, units="m")
        self.add_input("spacing_primary", spacing_primary, units="unitless")
        self.add_input("spacing_secondary", spacing_secondary, units="unitless")
        self.add_input("angle_orientation", angle_orientation, units="deg")
        self.add_input("angle_skew", angle_skew, units="deg")

    def setup_partials(self):
        """Derivative setup for OM component."""

        # default FD for the layout tools
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """Computation for the OM component."""

        D_rotor = self.windIO["wind_farm"]["turbine"][
            "rotor_diameter"
        ]  # will break if multiple turbine types are used...
        lengthscale_spacing_streamwise = inputs["spacing_primary"] * D_rotor
        lengthscale_spacing_spanwise = inputs["spacing_secondary"] * D_rotor

        N_square = int(np.sqrt(self.N_turbines))  # floors

        count_y, count_x = np.meshgrid(
            np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2)),
            np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2)),
        )

        if self.N_turbines == N_square**2:
            pass
        elif self.N_turbines <= N_square * (N_square + 1):
            # grid farm is a little bit above the last square... add a trailing
            # row.
            count_x = np.vstack([count_x, ((N_square + 1) / 2) * np.ones((N_square,))])
            count_y = np.vstack(
                [count_y, np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2))]
            )
            count_x = count_x.flatten()
            count_y = count_y.flatten()
        else:
            # grid farm is nearly the next square... oversize and cut the last
            count_y, count_x = np.meshgrid(
                np.arange(-((N_square) / 2), ((N_square + 2) / 2)),
                np.arange(-((N_square) / 2), ((N_square + 2) / 2)),
            )
        count_x = count_x.flatten()[: self.N_turbines]
        count_y = count_y.flatten()[: self.N_turbines]

        angle_skew = -np.pi / 180.0 * inputs["angle_skew"]
        Bmtx = np.array(
            [
                [1.0, 0.0],
                [np.tan(angle_skew[0]), 1.0],
            ]
        ).squeeze()

        xi_positions = count_x * lengthscale_spacing_spanwise
        yi_positions = count_y * lengthscale_spacing_streamwise

        angle_orientation = np.pi / 180.0 * inputs["angle_orientation"]
        Amtx = np.array(
            [
                [np.cos(angle_orientation), np.sin(angle_orientation)],
                [-np.sin(angle_orientation), np.cos(angle_orientation)],
            ]
        ).squeeze()
        xyp = Amtx @ (Bmtx @ np.vstack([xi_positions, yi_positions]))

        outputs["x_turbines"] = (xyp[0, :] + inputs["x0"]).tolist()
        outputs["y_turbines"] = (xyp[1, :] + inputs["y0"]).tolist()

        outputs["spacing_effective_primary"] = inputs["spacing_primary"]
        outputs["spacing_effective_secondary"] = np.sqrt(
            inputs["spacing_secondary"] ** 2.0 / np.cos(angle_skew) ** 2.0
        )


class GridFarmLanduse(templates.LanduseTemplate):
    """
    Landuse class for four-parameter parallelepiped grid farm layout.

    This is a class to compute the landuse area of the parameterized, structured
    grid farm defined in `GridFarmLayout`.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from
        `templates.LayoutTemplate`)
    N_turbines : int
        the number of turbines that should be in the farm layout (inherited from
        `templates.LayoutTemplate`)

    Inputs
    ------
    distance_layback_diameters : float
        the number of diameters of layback desired for the landuse calculation
        (inherited from `templates.LayoutTemplate`)
    angle_orientation : float
        orientation in degrees clockwise with respect to North of the primary
        axis of the wind farm layout
    spacing_primary : float
        spacing of turbine rows along the primary axis (rotated by
        `angle_orientation`) in nondimensional rotor diameters
    spacing_secondary : float
        spacing of turbine columns along the perpendicular to the primary axis
        (rotated by 90° with respect to the primary axis) in nondimensional
        rotor diameters
    angle_skew : float
        clockwise skew angle of turbine rows w.r.t. beyond the 90° clockwise
        perpendicular to the primary axis

    Outputs
    -------
    area_tight : float
        the area in square kilometers that the farm occupies based on the
        circumscribing geometry with a specified (default zero) layback buffer
        (inherited from `templates.LayoutTemplate`)
    area_aligned_parcel : float
        the area in square kilometers that the farm occupies based on the
        circumscribing rectangle that is aligned with the primary axis of the
        wind farm plus a specified (default zero) layback buffer
    area_compass_parcel : float
        the area in square kilometers that the farm occupies based on the
        circumscribing rectangle that is aligned with the compass rose plus a
        specified (default zero) layback buffer
    """

    def setup(self):
        """Setup of OM component."""

        super().setup()

        # add grid farm-specific inputs
        self.add_input(
            "spacing_primary",
            self.modeling_options["layout"]["spacing_primary"],
            units="unitless",
            desc="turbine row spacing in rotor diameters",
        )
        self.add_input(
            "spacing_secondary",
            self.modeling_options["layout"]["spacing_secondary"],
            units="unitless",
            desc="turbine column spacing (along rows) in rotor diameters",
        )
        self.add_input(
            "angle_orientation",
            self.modeling_options["layout"]["angle_orientation"],
            units="deg",
            desc="orientation in degrees clockwise with respect to North",
        )
        self.add_input(
            "angle_skew",
            self.modeling_options["layout"]["angle_skew"],
            units="deg",
            desc="clockwise skew angle of turbine rows",
        )

        self.add_output(
            "area_aligned_parcel",
            0.0,
            units="km**2",
            desc="area of the tightest rectangle around the farm (plus layback) that is aligned with the orientation vector",
        )
        self.add_output(
            "area_compass_parcel",
            0.0,
            units="km**2",
            desc="area of the tightest rectangular and compass-aligned land parcel that will fit the farm (plus layback)",
        )

    def setup_partials(self):
        """Derivative setup for OM component."""

        # default complex step for the layout tools, since they're often algebraic
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        """Computation for the OM component."""

        D_rotor = self.windIO["wind_farm"]["turbine"][
            "rotor_diameter"
        ]  # will break if multiple turbine types are used...
        lengthscale_spacing_streamwise = inputs["spacing_primary"] * D_rotor
        lengthscale_spacing_spanwise = inputs["spacing_secondary"] * D_rotor
        lengthscale_layback = inputs["distance_layback_diameters"] * D_rotor

        N_square = int(np.sqrt(self.N_turbines))  # floors

        min_count_xf = -(N_square - 1) / 2
        min_count_yf = -(N_square - 1) / 2
        max_count_xf = (N_square - 1) / 2
        max_count_yf = (N_square - 1) / 2

        if self.N_turbines == N_square**2:
            pass
        elif self.N_turbines <= N_square * (N_square + 1):
            max_count_xf = (N_square + 1) / 2
        else:
            min_count_xf = -(N_square) / 2
            min_count_yf = -(N_square) / 2
            max_count_xf = (N_square) / 2
            max_count_yf = (N_square) / 2

        # the side lengths of a parallelopiped oriented with the farm that encloses the farm with layback
        length_farm_xf = (max_count_xf - min_count_xf) * lengthscale_spacing_spanwise
        length_farm_yf = (max_count_yf - min_count_yf) * lengthscale_spacing_streamwise

        # the area of a parallelopiped oriented with the farm that encloses the farm with layback
        area_parallelopiped = (length_farm_xf + 2 * lengthscale_layback) * (
            length_farm_yf + 2 * lengthscale_layback
        )

        # the side lengths of a square oriented with the farm that encloses the farm with layback
        angle_skew = np.pi / 180.0 * inputs["angle_skew"]
        length_enclosing_farm_xf = length_farm_xf
        length_enclosing_farm_yf = (
            max_count_yf * lengthscale_spacing_streamwise
            + np.abs(max_count_xf)
            * lengthscale_spacing_spanwise
            * np.abs(np.tan(angle_skew))
        ) - (
            min_count_yf * lengthscale_spacing_streamwise
            - np.abs(min_count_xf)
            * lengthscale_spacing_spanwise
            * np.abs(np.tan(angle_skew))
        )

        # the area of a square oriented with the farm that encloses the farm with layback
        area_enclosingsquare_farmoriented = (
            length_enclosing_farm_xf + 2 * lengthscale_layback
        ) * (length_enclosing_farm_yf + 2 * lengthscale_layback)

        # the side lengths of a square oriented with the compass rose that encloses the farm with layback
        angle_orientation = np.pi / 180.0 * inputs["angle_orientation"]
        A_x = (
            +max_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
            + max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
            - (max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
            * np.tan(angle_skew)
        )
        A_y = (
            +max_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
            - max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
            - (max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
            * np.tan(angle_skew)
        )
        B_x = (
            +min_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
            + max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
            - (max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
            * np.tan(angle_skew)
        )
        B_y = (
            +min_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
            - max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
            - (max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
            * np.tan(angle_skew)
        )
        C_x = (
            min_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
            + min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
            - (min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
            * np.tan(angle_skew)
        )
        C_y = (
            min_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
            - min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
            - (min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
            * np.tan(angle_skew)
        )
        D_x = (
            -min_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
            + min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
            - (min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
            * np.tan(angle_skew)
        )
        D_y = (
            -min_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
            - min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
            - (min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
            * np.tan(angle_skew)
        )

        length_enclosing_farm_x = np.max([A_x, B_x, C_x, D_x]) - np.min(
            [A_x, B_x, C_x, D_x]
        )
        length_enclosing_farm_y = np.max([A_y, B_y, C_y, D_y]) - np.min(
            [A_y, B_y, C_y, D_y]
        )

        area_enclosingsquare_compass = (
            length_enclosing_farm_x + 2 * lengthscale_layback
        ) * (length_enclosing_farm_y + 2 * lengthscale_layback)
        outputs["area_tight"] = area_parallelopiped / (1e3) ** 2
        outputs["area_aligned_parcel"] = area_enclosingsquare_farmoriented / (1e3) ** 2
        outputs["area_compass_parcel"] = area_enclosingsquare_compass / (1e3) ** 2
