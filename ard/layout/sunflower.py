import numpy as np
import scipy.spatial

import ard.layout.templates as templates
import ard.layout.fullfarm as fullfarm

phi_golden = (1 + np.sqrt(5)) / 2  # golden ratio


def sunflower(
    n: float,
    alpha: float = 0,  # proportion of points that should end on boundary
    n_b: float = None,  # for overriding with the number of boundary elements
    geodesic=False,  # use geodesic step function
):
    """
    generate a sunflower seed packing pattern

    adapted from a stackoverflow post:
        https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle#28572551

    in turn from the wolfram demonstrations page:
        Joost de Jong (2013),
        "Sunflower Seed Arrangements"
        Wolfram Demonstrations Project.
        demonstrations.wolfram.com/SunflowerSeedArrangements/.

    appears to originate from: doi:10.1016/0025-5564(79)90080-4
    """

    def radius(k: int, n: int, b: int):
        """
        radius at which a seed should live
        b sets the number of boundary points
        remainder of n points have sequence-baased location
        """

        if k > n - b:
            return 1.0
        else:
            return np.sqrt(k - 0.5) / np.sqrt(n - (b + 1) / 2)

    points = []  # initialize a set of points
    # each next angle should step by a certain amount
    angle_stride = 2 * np.pi * phi_golden if geodesic else 2 * np.pi / phi_golden**2
    b = (
        n_b if n_b is not None else round(alpha * np.sqrt(n))
    )  # number of boundary points
    for k in range(1, n + 1):
        r = radius(k, n, b)  # get radius
        theta = k * angle_stride  # get angle
        points.append((r * np.cos(theta), r * np.sin(theta)))
    return points


class SunflowerFarmLayout(templates.LayoutTemplate):
    """
    A sunflower-inspired structured layout algorithm

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
    alpha : float
        a parameter to control the number of boundary (v. interior) turbines
    spacing_target : float
        a parameter to control the target average minimum spacing

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

        # add parameters for sunflower farm DVs
        # self.add_input("alpha", 0.0, desc="boundary point control param.")
        self.add_input("spacing_target", 0.0, desc="target spacing in rotor diameters")

    def setup_partials(self):
        """Derivative setup for OM component."""

        # run FD for the layout tools
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """Computation for the OM component."""

        # get the desired mean nearest-neighbor distance
        D_rotor = self.windIO["wind_farm"]["turbine"][
            "rotor_diameter"
        ]  # get rotor diameter
        spacing_target = D_rotor * inputs["spacing_target"]

        # generate the points
        points = np.array(sunflower(self.N_turbines, geodesic=True))
        dist_mtx = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(points)
        )
        np.fill_diagonal(dist_mtx, np.inf)  # self-distance not meaningful, remove
        d_mean_NN = np.mean(np.min(dist_mtx, axis=0))

        # rescale points to achieve target spacing
        points *= spacing_target / d_mean_NN

        # pipe in the outputs
        outputs["x_turbines"] = points[:, 0].tolist()
        outputs["y_turbines"] = points[:, 1].tolist()
        outputs["spacing_effective_primary"] = spacing_target  # ???
        outputs["spacing_effective_secondary"] = spacing_target  # ???


class SunflowerFarmLanduse(fullfarm.FullFarmLanduse):
    pass  # just use the full-farm landuse component
