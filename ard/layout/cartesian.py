import numpy as np
from scipy.spatial.distance import cdist

import ard.layout.templates as templates
import ard.layout.fullfarm as fullfarm


class CartesianLayout(templates.LayoutTemplate):
    """
    A layout class that reads explicit Cartesian coordinates from config.

    This layout type reads x_turbines and y_turbines directly from
    modeling_options.layout and outputs them. Positions are not generated
    from spacing parameters - they are fully specified in the configuration.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary containing x_turbines and y_turbines lists

    Inputs
    ------
    None - layout is fully specified in config

    Outputs
    -------
    x_turbines : np.ndarray
        x-coordinates of turbines from modeling_options
    y_turbines : np.ndarray
        y-coordinates of turbines from modeling_options
    yaw_turbines : np.ndarray
        yaw angles (degrees) of turbines from modeling_options
    spacing_effective_primary : float
        approximate primary spacing for BOS calculation
    spacing_effective_secondary : float
        approximate secondary spacing for BOS calculation
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

    def setup(self):
        """Setup of OM component."""
        super().setup()

    def setup_partials(self):
        """Derivative setup for OM component."""
        # No inputs, so no partials to declare
        pass

    def compute(self, inputs, outputs):
        """Computation for the OM component."""
        # Get the x, y, and yaw coordinates from modeling_options
        layout_options = self.modeling_options["layout"]
        x_turbines = np.array(layout_options.get("x_turbines", []))
        y_turbines = np.array(layout_options.get("y_turbines", []))
        yaw_turbines = np.array(layout_options.get("yaw_turbines", [0.0] * self.N_turbines))

        if len(x_turbines) != self.N_turbines or len(y_turbines) != self.N_turbines:
            raise ValueError(
                f"Cartesian layout: x_turbines and y_turbines must have length {self.N_turbines}, "
                f"got {len(x_turbines)} and {len(y_turbines)}"
            )

        if len(yaw_turbines) != self.N_turbines:
            raise ValueError(
                f"Cartesian layout: yaw_turbines must have length {self.N_turbines}, "
                f"got {len(yaw_turbines)}"
            )

        outputs["x_turbines"] = x_turbines
        outputs["y_turbines"] = y_turbines
        outputs["yaw_turbines"] = yaw_turbines

        # Compute effective spacing from the actual layout
        points = np.column_stack([x_turbines, y_turbines])
        if self.N_turbines > 1:
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            mean_nearest_neighbor = np.mean(np.min(distances, axis=1))
            D_rotor = self.windIO["wind_farm"]["turbine"]["rotor_diameter"]
            outputs["spacing_effective_primary"] = mean_nearest_neighbor / D_rotor
            outputs["spacing_effective_secondary"] = mean_nearest_neighbor / D_rotor
        else:
            outputs["spacing_effective_primary"] = 0.0
            outputs["spacing_effective_secondary"] = 0.0


class CartesianFarmLanduse(fullfarm.FullFarmLanduse):
    """Landuse component for Cartesian layout."""
    pass
