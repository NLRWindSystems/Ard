import numpy as np
from scipy.spatial.distance import cdist

import ard.layout.templates as templates


class FreeLayout(templates.LayoutTemplate):
    """Layout component that takes x/y turbine positions as OpenMDAO inputs.

    Unlike CartesianLayout (which reads positions from modeling_options config),
    FreeLayout takes x_turbines_in and y_turbines_in as OpenMDAO inputs so they
    can be driven by an outer IVC and optimized via gradient-based methods.

    The outer driver sets positions through the IVC -> SubmodelComp boundary.
    Initial values are read from modeling_options so the model has sensible
    defaults before optimization starts.

    Options
    -------
    modeling_options : dict
        Modeling options dict; x_turbines / y_turbines used for initial values.

    Inputs
    ------
    x_turbines_in : np.ndarray
        x-coordinates of turbines (meters). Driven from outer IVC.
    y_turbines_in : np.ndarray
        y-coordinates of turbines (meters). Driven from outer IVC.

    Outputs
    -------
    x_turbines, y_turbines, yaw_turbines : same as LayoutTemplate
    spacing_effective_primary, spacing_effective_secondary : computed from positions
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        super().setup()  # adds x_turbines, y_turbines, yaw_turbines, spacing_effective_* outputs

        layout = self.modeling_options["layout"]
        n = self.N_turbines
        x0 = np.array(layout.get("x_turbines", np.zeros(n)), dtype=float)
        y0 = np.array(layout.get("y_turbines", np.zeros(n)), dtype=float)

        # Use distinct input names to avoid conflict with the promoted output names
        self.add_input("x_turbines_in", val=x0, shape=n, units="m",
                       desc="turbine x-coordinates from outer IVC")
        self.add_input("y_turbines_in", val=y0, shape=n, units="m",
                       desc="turbine y-coordinates from outer IVC")

    def setup_partials(self):
        n = self.N_turbines
        idx = np.arange(n)
        # Pass-through identity: analytic sparse declarations so the gradient
        # chain x_turbines_in → x_turbines → FLOWFarm is correctly traced.
        self.declare_partials("x_turbines", "x_turbines_in", rows=idx, cols=idx, val=np.ones(n))
        self.declare_partials("y_turbines", "y_turbines_in", rows=idx, cols=idx, val=np.ones(n))
        # spacing_effective_* depend non-trivially on both x and y positions; use FD.
        self.declare_partials("spacing_effective_primary",
                              ["x_turbines_in", "y_turbines_in"], method="fd")
        self.declare_partials("spacing_effective_secondary",
                              ["x_turbines_in", "y_turbines_in"], method="fd")
        # yaw_turbines is constant zero — no dependence on positions.

    def compute(self, inputs, outputs):
        x = inputs["x_turbines_in"]
        y = inputs["y_turbines_in"]
        n = len(x)

        outputs["x_turbines"] = x
        outputs["y_turbines"] = y
        outputs["yaw_turbines"] = np.zeros(n)

        if n > 1:
            points = np.column_stack([x, y])
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            mean_nn = np.mean(np.min(distances, axis=1))
            D_rotor = self.windIO["wind_farm"]["turbine"]["rotor_diameter"]
            spacing = mean_nn / D_rotor
        else:
            spacing = 0.0

        outputs["spacing_effective_primary"] = spacing
        outputs["spacing_effective_secondary"] = spacing
