from pathlib import Path
import pytest

import openmdao.api as om

import ard
import numpy as np
import ard.utils.io
import ard.layout.gridfarm as gridfarm
import ard.collection
import ard.cost.orbit_wrap as ocost


@pytest.mark.usefixtures("subtests")
class TestORBITNoApproxBranch:

    def test_raise_error(self):

        # specify the configuration/specification files to use
        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-22MW-284m-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # set up the modeling options
        windIO_plant = {
            "site": {
                "energy_resource": {
                    "wind_resource": {
                        "shear": 0.2,
                    },
                },
            },
            "wind_farm": {
                "turbine": ard.utils.io.load_yaml(filename_turbine),
                "electrical_substations": [
                    {
                        "electrical_substation": {
                            "coordinates": {
                                "x": [100.0],
                                "y": [100.0],
                            },
                        },
                    }
                ],
            },
        }
        modeling_options = {
            "windIO_plant": windIO_plant,
            "layout": {
                "N_turbines": 7,
                "N_substations": 1,
                "x_turbines": np.linspace(-7.0 * 400.0, 7.0 * 400.0, 7),
                "y_turbines": np.linspace(7.0 * 400.0, -7.0 * 400.0, 7),
            },
            "offshore": True,
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
            "collection": {
                "max_turbines_per_string": 8,
                "model_options": dict(
                    topology="branched",
                    feeder_route="segmented",
                    feeder_limit="unlimited",
                ),
                "solver_name": "highs",
                "solver_options": dict(
                    time_limit=10,
                    mip_gap=0.005,  # TODO ???
                ),
            },
            "costs": {
                "num_blades": 3,
                "tower_length": 149.386,
                "tower_mass": 1574044.87111,
                "nacelle_mass": 849143.2357,
                "blade_mass": 83308.31171,
                "turbine_capex": 1397.17046735,
                "site_mean_windspeed": 10.0,  # (m/s)
                "turbine_rated_windspeed": 11.13484394,  # (m/s)
                "commissioning_cost_kW": 44.0,  # (USD/kW)
                "decommissioning_cost_kW": 58.0,  # (USD/kW)
                "plant_substation_distance": 1.0,  # (km)
                "interconnection_distance": 8.5,  # (km)
                "site_distance": 115.0,  # (km)
                "site_distance_to_landfall": 50.0,  # (km)
                "port_cost_per_month": 2000000.0,  # (USD/mo)
                "construction_insurance": 44.0,  # (USD/kW)
                "construction_financing": 183.0,  # (USD/kW)
                "contingency": 316.0,  # (USD/kW)
                "site_auction_price": 100000000.0,  # (USD)
                "site_assessment_cost": 50000000.0,  # (USD)
                "construction_plan_cost": 250000.0,  # (USD)
                "installation_plan_cost": 1000000.0,  # (USD)
                "boem_review_cost": 0.0,  # (USD)
                "transition_piece_mass": 100.0e3,  # (kg)
                "transition_piece_cost": 0.0,  # (USD)
                # # Fixed bottom configuration
                # monopile_mass: 2097.21115974 # (tonne)
                # monopile_cost: 4744119.28172591 # (USD)
                # tcc_per_kW: 1397.17046735 # (USD/kW)
                # opex_per_kW: 110. # (USD/kW)
                # # Floating configuration
                "num_mooring_lines": 3,  # (-)
                "mooring_line_mass": 843225.1875,  # (kg)
                "mooring_line_diameter": 0.225,  # (m)
                "mooring_line_length": 837.0,  # (m)
                "anchor_mass": 0.0,  # (kg)
                "floating_substructure_cost": 11803978.242949858,  # (USD)
            },
        }

        # create an OM model and problem
        model = om.Group()
        coll = model.add_subsystem(  # collection component
            "collection",
            ard.collection.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )

        orbit = model.add_subsystem(
            "orbit",
            ocost.ORBITDetailedGroup(
                modeling_options=modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )
        model.connect("collection.terse_links", "orbit.terse_links")

        model.set_input_defaults(
            "x_turbines", modeling_options["layout"]["x_turbines"], units="km"
        )
        model.set_input_defaults(
            "y_turbines", modeling_options["layout"]["y_turbines"], units="km"
        )
        model.set_input_defaults(
            "x_substations",
            [
                li["electrical_substation"]["coordinates"]["x"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )
        model.set_input_defaults(
            "y_substations",
            [
                li["electrical_substation"]["coordinates"]["y"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )

        prob = om.Problem(model)
        prob.setup()

        prob.set_val("x_turbines", modeling_options["layout"]["x_turbines"], units="m")
        prob.set_val("y_turbines", modeling_options["layout"]["y_turbines"], units="m")

        prob.set_val(
            "x_substations",
            [
                li["electrical_substation"]["coordinates"]["x"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )
        prob.set_val(
            "y_substations",
            [
                li["electrical_substation"]["coordinates"]["y"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )

        # this configuration should not work
        with pytest.raises(ValueError):
            prob.run_model()

    def test_baseline_farm(self, subtests):

        # specify the configuration/specification files to use
        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-22MW-284m-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # set up the modeling options
        windIO_plant = {
            "site": {
                "energy_resource": {
                    "wind_resource": {
                        "shear": 0.2,
                    },
                },
            },
            "wind_farm": {
                "turbine": ard.utils.io.load_yaml(filename_turbine),
                "electrical_substations": [
                    {
                        "electrical_substation": {
                            "coordinates": {
                                "x": [100.0],
                                "y": [100.0],
                            },
                        },
                    }
                ],
            },
        }
        modeling_options = {
            "windIO_plant": windIO_plant,
            "layout": {
                "N_turbines": 7,
                "N_substations": 1,
                "x_turbines": np.linspace(-7.0 * 400.0, 7.0 * 400.0, 7),
                "y_turbines": np.linspace(7.0 * 400.0, -7.0 * 400.0, 7),
            },
            "offshore": True,
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
            "collection": {
                "max_turbines_per_string": 8,
                "model_options": dict(
                    topology="radial",
                    feeder_route="segmented",
                    feeder_limit="unlimited",
                ),
                "solver_name": "highs",
                "solver_options": dict(
                    time_limit=10,
                    mip_gap=0.005,  # TODO ???
                ),
            },
            "costs": {
                "rated_power": 22000000.0,  # (W)
                "num_blades": 3,
                "tower_length": 149.386,  # (m)
                "tower_mass": 1574044.87111,  # (tonne)
                "nacelle_mass": 849143.2357,  # (tonne)
                "blade_mass": 83308.31171,  # (tonne)
                "turbine_capex": 1397.17046735,  # (tonne)
                "site_mean_windspeed": 10.0,  # (m/s)
                "turbine_rated_windspeed": 11.13484394,  # (m/s)
                "commissioning_cost_kW": 44.0,  # (USD/kW)
                "decommissioning_cost_kW": 58.0,  # (USD/kW)
                "plant_substation_distance": 1.0,  # (km)
                "interconnection_distance": 8.5,  # (km)
                "site_distance": 115.0,  # (km)
                "site_distance_to_landfall": 50.0,  # (km)
                "port_cost_per_month": 2000000.0,  # (USD/mo)
                "construction_insurance": 44.0,  # (USD/kW)
                "construction_financing": 183.0,  # (USD/kW)
                "contingency": 316.0,  # (USD/kW)
                "site_auction_price": 100000000.0,  # (USD)
                "site_assessment_cost": 50000000.0,  # (USD)
                "construction_plan_cost": 250000.0,  # (USD)
                "installation_plan_cost": 1000000.0,  # (USD)
                "boem_review_cost": 0.0,  # (USD)
                "transition_piece_mass": 100.0e3,  # (kg)
                "transition_piece_cost": 0.0,  # (USD)
                # # Fixed bottom configuration
                # monopile_mass: 2097.21115974 # (tonne)
                # monopile_cost: 4744119.28172591 # (USD)
                # tcc_per_kW: 1397.17046735 # (USD/kW)
                # opex_per_kW: 110. # (USD/kW)
                # # Floating configuration
                "num_mooring_lines": 3,  # (-)
                "mooring_line_mass": 843225.1875,  # (kg)
                "mooring_line_diameter": 0.225,  # (m)
                "mooring_line_length": 837.0,  # (m)
                "anchor_mass": 0.0,  # (kg)
                "floating_substructure_cost": 11803978.242949858,  # (USD)
            },
        }

        # create an OM model and problem
        model = om.Group()
        coll = model.add_subsystem(  # collection component
            "collection",
            ard.collection.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )

        orbit = model.add_subsystem(
            "orbit",
            ocost.ORBITDetailedGroup(
                modeling_options=modeling_options,
                approximate_branches=False,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )
        model.connect("collection.terse_links", "orbit.terse_links")

        model.set_input_defaults(
            "x_turbines", modeling_options["layout"]["x_turbines"], units="km"
        )
        model.set_input_defaults(
            "y_turbines", modeling_options["layout"]["y_turbines"], units="km"
        )
        model.set_input_defaults(
            "x_substations",
            [
                li["electrical_substation"]["coordinates"]["x"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )
        model.set_input_defaults(
            "y_substations",
            [
                li["electrical_substation"]["coordinates"]["y"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )

        prob = om.Problem(model)
        prob.setup()

        prob.set_val("x_turbines", modeling_options["layout"]["x_turbines"], units="m")
        prob.set_val("y_turbines", modeling_options["layout"]["y_turbines"], units="m")

        prob.set_val(
            "x_substations",
            [
                li["electrical_substation"]["coordinates"]["x"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )
        prob.set_val(
            "y_substations",
            [
                li["electrical_substation"]["coordinates"]["y"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )

        prob.run_model()

        bos_capex = float(prob.get_val("orbit.bos_capex", units="MUSD"))
        total_capex = float(prob.get_val("orbit.total_capex", units="MUSD"))

        bos_capex_ref = 477.3328175080761
        total_capex_ref = 727.9128175080762

        with subtests.test(f"orbit_skew_bos"):
            assert np.isclose(bos_capex, bos_capex_ref, rtol=1e-3)
        with subtests.test(f"orbit_skew_total"):
            assert np.isclose(total_capex, total_capex_ref, rtol=1e-3)


@pytest.mark.usefixtures("subtests")
class TestORBITApproxBranch:

    def setup_method(self):

        # specify the configuration/specification files to use
        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-22MW-284m-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # set up the modeling options
        windIO_plant = self.windIO_plant = {
            "site": {
                "energy_resource": {
                    "wind_resource": {
                        "shear": 0.2,
                    },
                },
            },
            "wind_farm": {
                "turbine": ard.utils.io.load_yaml(filename_turbine),
                "electrical_substations": [
                    {
                        "electrical_substation": {
                            "coordinates": {
                                "x": [100.0],
                                "y": [100.0],
                            },
                        },
                    }
                ],
            },
        }
        modeling_options = self.modeling_options = {
            "windIO_plant": windIO_plant,
            "layout": {
                "N_turbines": 7,
                "N_substations": 1,
                "x_turbines": np.linspace(-7.0 * 400.0, 7.0 * 400.0, 7),
                "y_turbines": np.linspace(7.0 * 400.0, -7.0 * 400.0, 7),
            },
            "offshore": True,
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
            "collection": {
                "max_turbines_per_string": 8,
                "model_options": dict(
                    topology="radial",
                    feeder_route="segmented",
                    feeder_limit="unlimited",
                ),
                "solver_name": "highs",
                "solver_options": dict(
                    time_limit=10,
                    mip_gap=0.005,  # TODO ???
                ),
            },
            "costs": {
                "rated_power": 22000000.0,  # W
                "num_blades": 3,
                "tower_length": 149.386,
                "tower_mass": 1574044.87111,
                "nacelle_mass": 849143.2357,
                "blade_mass": 83308.31171,
                "turbine_capex": 1397.17046735,
                "site_mean_windspeed": 10.0,  # (m/s)
                "turbine_rated_windspeed": 11.13484394,  # (m/s)
                "commissioning_cost_kW": 44.0,  # (USD/kW)
                "decommissioning_cost_kW": 58.0,  # (USD/kW)
                "plant_substation_distance": 1.0,  # (km)
                "interconnection_distance": 8.5,  # (km)
                "site_distance": 115.0,  # km
                "site_distance_to_landfall": 50.0,  # km
                "port_cost_per_month": 2000000.0,  # (USD/mo)
                "construction_insurance": 44.0,  # (USD/kW)
                "construction_financing": 183.0,  # (USD/kW)
                "contingency": 316.0,  # (USD/kW)
                "site_auction_price": 100000000.0,  # (USD)
                "site_assessment_cost": 50000000.0,  # (USD)
                "construction_plan_cost": 250000.0,  # (USD)
                "installation_plan_cost": 1000000.0,  # (USD)
                "boem_review_cost": 0.0,  # (USD)
                "transition_piece_mass": 100.0e3,  # kg
                "transition_piece_cost": 0.0,  # (USD)
                # # Fixed bottom configuration
                # monopile_mass: 2097.21115974 # (tonne)
                # monopile_cost: 4744119.28172591 # (USD)
                # tcc_per_kW: 1397.17046735 # (USD/kW)
                # opex_per_kW: 110. # (USD/kW)
                # # Floating configuration
                "num_mooring_lines": 3,  # (-)
                "mooring_line_mass": 843225.1875,  # (kg)
                "mooring_line_diameter": 0.225,  # (m)
                "mooring_line_length": 837.0,  # (m)
                "anchor_mass": 0.0,  # (kg)
                "floating_substructure_cost": 11803978.242949858,  # (USD)
            },
        }

        # create an OM model and problem
        self.model = om.Group()
        self.coll = self.model.add_subsystem(  # collection component
            "collection",
            ard.collection.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )

        self.orbit = self.model.add_subsystem(
            "orbit",
            ocost.ORBITDetailedGroup(
                modeling_options=self.modeling_options,
                approximate_branches=True,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )
        self.model.connect("collection.terse_links", "orbit.terse_links")

        self.model.set_input_defaults(
            "x_turbines", self.modeling_options["layout"]["x_turbines"], units="km"
        )
        self.model.set_input_defaults(
            "y_turbines", self.modeling_options["layout"]["y_turbines"], units="km"
        )
        self.model.set_input_defaults(
            "x_substations",
            [
                li["electrical_substation"]["coordinates"]["x"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )
        self.model.set_input_defaults(
            "y_substations",
            [
                li["electrical_substation"]["coordinates"]["y"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_baseline_farm(self, subtests):

        self.prob.set_val(
            "x_turbines", self.modeling_options["layout"]["x_turbines"], units="m"
        )
        self.prob.set_val(
            "y_turbines", self.modeling_options["layout"]["y_turbines"], units="m"
        )

        self.prob.set_val(
            "x_substations",
            [
                li["electrical_substation"]["coordinates"]["x"]
                for li in self.windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )
        self.prob.set_val(
            "y_substations",
            [
                li["electrical_substation"]["coordinates"]["y"]
                for li in self.windIO_plant["wind_farm"]["electrical_substations"]
            ],
            units="km",
        )

        self.prob.run_model()

        bos_capex = float(self.prob.get_val("orbit.bos_capex", units="MUSD"))
        total_capex = float(self.prob.get_val("orbit.total_capex", units="MUSD"))

        bos_capex_ref = 477.3328175080761
        total_capex_ref = 727.9128175080762

        with subtests.test(f"orbit_skew_bos"):
            assert np.isclose(bos_capex, bos_capex_ref, rtol=1e-3)
        with subtests.test(f"orbit_skew_total"):
            assert np.isclose(total_capex, total_capex_ref, rtol=1e-3)
