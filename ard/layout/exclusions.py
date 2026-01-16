import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
import ard.utils.geometry
import openmdao.api as om


class FarmExclusionDistancePolygon(om.ExplicitComponent):
    """
    A class to return distances between turbines and a polygonal exclusion, or
    sets of polygonal exclusion regions.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.windIO = self.modeling_options["windIO_plant"]
        self.N_turbines = int(self.modeling_options["layout"]["N_turbines"])

        # load exclusion vertices from windIO file
        if "exclusions" not in self.windIO["site"]:
            raise KeyError(
                "You have requested an exclusion but no exclusions were found in the windIO file."
            )
        if "circle" in self.windIO["site"]["exclusions"]:
            raise NotImplementedError(
                "The circular exclusions from windIO have not been implemented here, yet."
            )
        if "polygons" not in self.windIO["site"]["exclusions"]:
            raise KeyError(
                "Currently only polygon exclusions from windIO have been implemented and none were found."
            )
        self.exclusion_vertices = [
            np.array(
                [
                    polygon["x"],
                    polygon["y"],
                ]
            ).T
            for polygon in self.windIO["site"]["exclusions"]["polygons"]
        ]
        self.exclusion_regions = self.modeling_options.get("exclusions", {}).get(
            "turbine_exclusion_assignments",  # get the exclusion region assignments from modeling_options, if there
            np.zeros(self.N_turbines, dtype=int),  # default to zero for all turbines
        )

        # prep the jacobian
        self.distance_multi_point_to_multi_polygon_ray_casting_jac = jax.jacfwd(
            ard.utils.geometry.distance_multi_point_to_multi_polygon_ray_casting, [0, 1]
        )

        # set up inputs and outputs for turbine exclusion distances
        self.add_input(
            "x_turbines", jnp.zeros((self.N_turbines,)), units="m"
        )  # x location of the turbines in m w.r.t. reference coordinates
        self.add_input(
            "y_turbines", jnp.zeros((self.N_turbines,)), units="m"
        )  # y location of the turbines in m w.r.t. reference coordinates

        self.add_output(
            "exclusion_distances",
            jnp.zeros(self.N_turbines),
            units="m",
        )

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(self.N_turbines),
            cols=np.arange(self.N_turbines),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]

        exclusion_distances = (
            ard.utils.geometry.distance_multi_point_to_multi_polygon_ray_casting(
                x_turbines,
                y_turbines,
                boundary_vertices=self.exclusion_vertices,
                regions=self.exclusion_regions,
            )
        )

        outputs["exclusion_distances"] = -exclusion_distances

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]

        jacobian = self.distance_multi_point_to_multi_polygon_ray_casting_jac(
            x_turbines, y_turbines, self.exclusion_vertices, self.exclusion_regions
        )

        partials["exclusion_distances", "x_turbines"] = -jacobian[0].diagonal()
        partials["exclusion_distances", "y_turbines"] = -jacobian[1].diagonal()
