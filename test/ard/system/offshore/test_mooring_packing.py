from pathlib import Path

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals

import pytest

import ard
import ard.utils.io
import ard.layout.gridfarm


@pytest.mark.usefixtures("subtests")
class TestMooringPacking:

    def setup_method(self):

        # specify the configuration/specification files to use
        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "ard"
            / "data"
            / "windIO-plant_turbine_IEA-22MW-284m-RWT.yaml"
        )  # windIO turbine specification
        data_windIO_turbine = ard.utils.io.load_yaml(filename_turbine)

        # set up the modeling options
        self.modeling_options = {
            "windIO_plant": {
                "name": "system test special",
                "site": {
                    "name": "system test site",
                    "boundaries": {
                        "polygons": [
                            {
                                "x": [-2.0, 2.0, 2.0, -2.0],
                                "y": [-2.0, -2.0, 2.0, 2.0],
                            },
                        ],
                    },
                },
                "wind_farm": {
                    "turbine": data_windIO_turbine,
                    "electrical_substations": [
                        {
                            "electrical_substation": {
                                "coordinates": {
                                    "x": [500.0],
                                    "y": [500.0],
                                },
                            },
                        },
                    ],
                },
            },
            "layout": {
                "N_turbines": 4,
                "N_substations": 1,
                "spacing_primary": 7.0,
                "spacing_secondary": 7.0,
                "angle_orientation": 0.0,
                "angle_skew": 0.0,
                "phi_platform": 0.0,
                "x_turbines": np.zeros(4),
                "y_turbines": np.zeros(4),
            },
            "offshore": False,
            "floating": False,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
            "collection": {
                "max_turbines_per_string": 8,
                "solver_name": "highs",
                "solver_options": dict(
                    time_limit=60,
                    mip_gap=0.005,  # TODO ???
                ),
                "model_options": dict(
                    topology="branched",
                    feeder_route="segmented",
                    feeder_limit="unlimited",
                ),
            },
        }

        # create the OpenMDAO model
        model = om.Group()

        model.add_subsystem(
            "layout",
            ard.layout.gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )

        model.add_subsystem(
            "landuse",
            ard.layout.gridfarm.GridFarmLanduse(modeling_options=self.modeling_options),
            promotes=["*"],
        )

        model.add_subsystem(
            "collection",
            ard.collection.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
            promotes=["x_turbines", "y_turbines"],
        )

        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_constant_depth.ConstantDepthMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["phi_platform", "x_turbines", "y_turbines"],
        )

        model.add_subsystem(  # regulatory constraints for mooring
            "mooring_constraint",
            ard.offshore.mooring_constraint.MooringConstraint(
                modeling_options=self.modeling_options,
            ),
            promotes=["x_turbines", "y_turbines"],
        )
        model.connect("mooring_design.x_anchors", "mooring_constraint.x_anchors")
        model.connect("mooring_design.y_anchors", "mooring_constraint.y_anchors")

        model.add_subsystem(  # constraints for turbine proximity
            "spacing_constraint",
            ard.layout.spacing.TurbineSpacing(
                modeling_options=self.modeling_options,
            ),
            promotes=["x_turbines", "y_turbines"],
        )

        # build out the problem based on this model
        self.prob = om.Problem(model)
        self.prob.setup()

    def test_packing(self):

        # add design variable
        self.prob.model.add_design_var("spacing_primary", lower=2.0, upper=15.0)
        self.prob.model.add_design_var("spacing_secondary", lower=2.0, upper=15.0)
        self.prob.model.add_design_var("angle_orientation", lower=-90.0, upper=90.0)
        self.prob.model.add_design_var("angle_skew", lower=-90.0, upper=90.0)
        self.prob.model.add_design_var("phi_platform", lower=-30.0, upper=30.0)
        self.prob.model.add_constraint(
            "mooring_constraint.mooring_spacing",
            units="m",
            lower=50.0,
        )
        self.prob.model.add_constraint(
            "spacing_constraint.turbine_spacing",
            units="m",
            lower=3
            * self.modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                "rotor_diameter"
            ],
        )
        self.prob.model.add_objective("collection.total_length_cables")

        # configure the driver
        self.prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
        self.prob.driver.options["maxiter"] = 50  # short run

        # setup the problem
        self.prob.setup()

        # run the model
        self.prob.run_model()

        # compute the derivatives
        totals = self.prob.compute_totals(
            of=[
                "mooring_constraint.mooring_spacing",
                "spacing_constraint.turbine_spacing",
            ],
            wrt=[
                "spacing_primary",
                "spacing_secondary",
                "angle_orientation",
                "angle_skew",
                "phi_platform",
            ],
        )

        # check total derivatives using OpenMDAO's check_totals and assert tools
        assert_check_totals(
            self.prob.check_totals(
                of=[
                    "mooring_constraint.mooring_spacing",
                    "spacing_constraint.turbine_spacing",
                ],
                wrt=[
                    "spacing_primary",
                    "spacing_secondary",
                    "angle_orientation",
                    "angle_skew",
                    "phi_platform",
                ],
                step=1e-6,
                form="central",
                show_only_incorrect=False,
                out_stream=None,
            )
        )


# FIN!
