import numpy as np

from shapely.geometry import Point
from shapely.ops import unary_union

import openmdao.api as om

_R_earth = 6371008.8  # Earth radius, m


def calculate_viewshed_section_angle(
    D_rotor: float,  # rotor diameter, m
    h_hub: float,  # hub height, m
    R_earth: float = _R_earth,  # Earth radius, m
    h_terrain: float = 0.0,  # mean height of prevailing terrain wrt turbine base, m
) -> float:  # returns arc angle (rad) of viewshed
    H = D_rotor / 2 + h_hub - h_terrain
    return np.arccos(R_earth / (R_earth + H))


def calculate_viewshed_arc_length(
    D_rotor: float,  # rotor diameter, m
    h_hub: float,  # hub height, m
    R_earth: float = _R_earth,  # Earth radius, m
    h_terrain: float = 0.0,  # mean height of prevailing terrain wrt turbine base, m
) -> float:  # returns arc length of viewshed
    angle_arc = calculate_viewshed_section_angle(
        D_rotor,
        h_hub,
        R_earth=R_earth,
        h_terrain=h_terrain,
    )
    return R_earth * angle_arc


def calculate_viewshed_arc_length_smallangle(
    D_rotor: float,  # rotor diameter, m
    h_hub: float,  # hub height, m
    R_earth: float = _R_earth,  # Earth radius, m
    h_terrain: float = 0.0,  # mean height of prevailing terrain wrt turbine base, m
) -> float:  # returns arc length of viewshed
    angle_arc = calculate_viewshed_section_angle(
        D_rotor,
        h_hub,
        R_earth=R_earth,
        h_terrain=h_terrain,
    )
    return R_earth * np.sin(angle_arc)


class ViewshedAreaComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        self.modeling_options = self.options["modeling_options"]
        self.windIO = self.modeling_options["windIO_plant"]

        self.D_rotor = self.windIO["wind_farm"]["turbine"]["rotor_diameter"]
        self.h_hub = self.windIO["wind_farm"]["turbine"]["hub_height"]
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]

        self.add_input("x_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y_turbines", np.zeros((self.N_turbines,)), units="m")
        # self.add_output('viewshed_arc_length', val=0.0, units='m', desc='Viewshed arc length')
        self.add_output("area_viewshed", val=0.0, units="km**2", desc="Viewshed area")

    def setup_partials(self):
        # declare FD because no derivative is available
        self.declare_partials("area_viewshed", "*", method="fd")

    def compute(self, inputs, outputs):

        # get the single-turbine viewshed arc length
        D_rotor = self.D_rotor
        h_hub = self.h_hub
        h_terrain = 0.0

        # project onto 2D surface plane
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        D_rotor_turbines = D_rotor * np.ones_like(x_turbines)
        h_hub_turbines = h_hub * np.ones_like(x_turbines)
        R_viewshed_turbines = calculate_viewshed_arc_length(
            D_rotor_turbines,
            h_hub_turbines,
            h_terrain=h_terrain,
        )

        # create a list of shapely circles
        viewshed_circles = [
            Point(x, y).buffer(r)
            for x, y, r in zip(
                x_turbines.flatten(),
                y_turbines.flatten(),
                R_viewshed_turbines.flatten(),
            )
        ]

        # compute the union of all circles
        viewshed_union = unary_union(viewshed_circles)

        # calculate the area of the union in square kilometers
        viewshed_union_area_km2 = viewshed_union.area / 1e6

        # pack and send output
        outputs["area_viewshed"] = viewshed_union_area_km2
