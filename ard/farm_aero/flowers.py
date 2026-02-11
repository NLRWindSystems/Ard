import os

import numpy as np
import pandas as pd

from ard.farm_aero.floris import create_FLORIS_turbine_from_windIO
from flowers import FlowersModel

import ard.farm_aero.templates as templates


class FLOWERSAEP(templates.FarmAEPTemplate):
    """
    Component class for computing an AEP analysis using FLOWERS.

    A component class that evaluates a series of farm power and associated
    quantities using FLOWERS with a wind rose to make an AEP estimate. Inherits
    the interface from `templates.FarmAEPTemplate`.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited via
        `templates.FarmAEPTemplate`)
    data_path: str
        not used for FLOWERS (inherited via `templates.FarmAEPTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (inherited via `templates.FarmAEPTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (inherited via `templates.FarmAEPTemplate`)
    yaw_turbines : np.ndarray
        not used for FLOWERS (inherited via `templates.FarmAEPTemplate`)

    Outputs
    -------
    AEP_farm : float
        the AEP of the farm given by the analysis (inherited from
        `templates.FarmAEPTemplate`)
    power_farm : np.ndarray
        an array of the farm power for each of the wind conditions that have
        been queried (inherited from `templates.FarmAEPTemplate`)
    # power_turbines : np.ndarray
    #     an array of the farm power for each of the turbines in the farm across
    #     all of the conditions that have been queried on the wind rose
    #     (`N_turbines`, `N_wind_conditions`) (inherited from
    #     `templates.FarmAEPTemplate`)
    # thrust_turbines : np.ndarray
    #     an array of the wind turbine thrust for each of the turbines in the farm
    #     across all of the conditions that have been queried on the wind rose
    #     (`N_turbines`, `N_wind_conditions`) (inherited from
    #     `templates.FarmAEPTemplate`)
    """

    def initialize(self):
        super().initialize()  # run super class script first!

    def setup(self):

        # override template to allow for input omission

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.windIO = self.modeling_options["windIO_plant"]
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]

        # set up inputs and outputs for farm layout
        self.add_input("x_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_output("AEP_farm", 0.0, units="W*h")

        # grab the windrose from the windIO data
        windrose_floris = templates.create_windresource_from_windIO(
            self.windIO,
            resource_type="probability",
        )
        windrose_resample = self.modeling_options.get("flowers", {}).get(
            "windrose_resample",
            self.modeling_options.get("wind_rose", {}).get("windrose_resample"),
        )
        if windrose_resample is not None:
            windrose_floris.resample_by_interpolation(
                **windrose_resample,
                inplace=True,
            )
        # extract to a dataframe
        self.wind_data = pd.DataFrame(
            {
                "wd": windrose_floris.wd_flat,
                "ws": windrose_floris.ws_flat,
                "freq_val": windrose_floris.freq_table_flat,
            }
        )

    def setup_partials(self):
        self.declare_partials("AEP_farm", ["x_turbines", "y_turbines"], method="exact")

    def compute(self, inputs, outputs):

        # extract the key inputs
        layout_x = inputs["x_turbines"]
        layout_y = inputs["y_turbines"]
        num_terms = self.modeling_options["flowers"]["num_terms"]
        k_wake_expansion = self.modeling_options["flowers"]["k"]

        # use the floris turbine as an intermediary to cover off windIO variants
        self.floris_turbine = create_FLORIS_turbine_from_windIO(self.windIO)

        rho_density_air = self.floris_turbine["power_thrust_table"][
            "ref_air_density"
        ]  # kg/m^3
        area_rotor = np.pi / 4 * self.floris_turbine["rotor_diameter"] ** 2  # m^2
        V_table = np.array(self.floris_turbine["power_thrust_table"]["wind_speed"])
        P_table = 1.0e3 * np.array(self.floris_turbine["power_thrust_table"]["power"])
        CT_table = np.array(
            self.floris_turbine["power_thrust_table"]["thrust_coefficient"]
        )
        CP_table = np.where(
            V_table == 0.0,
            0.0,
            P_table / (0.5 * rho_density_air * area_rotor * V_table**3),
        )

        turbine_type = {
            "D": self.windIO["wind_farm"]["turbine"]["rotor_diameter"],
            "U": self.windIO["wind_farm"]["turbine"]["performance"].get(
                "cutout_wind_speed", 25.0
            ),
            "ct": CT_table,
            "u_ct": V_table,
            "cp": CP_table,
            "u_cp": V_table,
        }

        # create the flowers model
        self.flowers_model = FlowersModel(
            self.wind_data,
            layout_x,
            layout_y,
            num_terms,
            k_wake_expansion,
            turbine_type,
        )

        # FLOWERS computes the powers
        outputs["AEP_farm"] = self.flowers_model.calculate_aep(
            rho_density=rho_density_air
        )
        # outputs["power_farm"] = FLOWERSFarmComponent.get_power_farm(self)
        # outputs["power_turbines"] = FLOWERSFarmComponent.get_power_turbines(self)
        # outputs["thrust_turbines"] = FLOWERSFarmComponent.get_thrust_turbines(self)

    def compute_partials(self, inputs, partials):

        # grab dependencies
        rho_density_air = self.floris_turbine["power_thrust_table"][
            "ref_air_density"
        ]  # kg/m^3

        # compute the gradients and extract to the right places
        _, gradient = self.flowers_model.calculate_aep(
            rho_density=rho_density_air, gradient=True
        )
        partials["AEP_farm", "x_turbines"] = gradient[:, 0]
        partials["AEP_farm", "y_turbines"] = gradient[:, 1]
