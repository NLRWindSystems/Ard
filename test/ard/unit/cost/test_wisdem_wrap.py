from pathlib import Path

import numpy as np
import openmdao.api as om

import ard
import ard.utils.io
import ard.utils.test_utils
import ard.layout.gridfarm as gridfarm
import ard.cost.wisdem_wrap as wcost


class TestLandBOSSE:

    def setup_method(self):

        # set up the modeling options
        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "ard"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        self.modeling_options = {
            "windIO_plant": {
                "site": {
                    "energy_resource": {
                        "wind_resource": {
                            "shear": 0.2,
                        },
                    },
                },
                "wind_farm": {
                    "turbine": ard.utils.io.load_yaml(filename_turbine),
                },
            },
            "layout": {
                "N_turbines": 25,
                "spacing_primary": 0.0,  # reset in test_setup
                "spacing_secondary": 0.0,  # reset in test_setup
                "angle_orientation": 0.0,  # reset in test_setup
                "angle_skew": 0.0,  # reset in test_setup
            },
            "costs": {
                "rated_power": 3400000.0,  # W
                "num_blades": 3,
                "rated_thrust_N": 645645.83964671,
                "gust_velocity_m_per_s": 52.5,
                "blade_surface_area": 69.7974979,
                "tower_mass": 620.4407337521,
                "nacelle_mass": 101.98582836439,
                "hub_mass": 8.38407517646,
                "blade_mass": 14.56341339641,
                "foundation_height": 0.0,
                "commissioning_cost_kW": 44.0,
                "decommissioning_cost_kW": 58.0,
                "trench_len_to_substation_km": 50.0,
                "distance_to_interconnect_mi": 4.97096954,
                "interconnect_voltage_kV": 130.0,
            },
        }

        # create an OM model and problem
        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.landbosse = self.model.add_subsystem(
            "landbosse",
            wcost.LandBOSSEGroup(modeling_options=self.modeling_options),
        )
        self.model.connect(  # effective primary spacing for BOS
            "spacing_effective_primary", "landbosse.turbine_spacing_rotor_diameters"
        )
        self.model.connect(  # effective secondary spacing for BOS
            "spacing_effective_secondary", "landbosse.row_spacing_rotor_diameters"
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_baseline_farm(self):

        self.prob.set_val("gridfarm.spacing_primary", 7.0)
        self.prob.set_val("gridfarm.spacing_secondary", 7.0)
        self.prob.set_val("gridfarm.angle_orientation", 0.0)
        self.prob.set_val("gridfarm.angle_skew", 0.0)

        self.prob.run_model()

        # use a file of pyrite-standard data to validate against
        fn_pyrite = Path(__file__).parent / "test_landbosse_wrap_baseline_farm.npz"
        test_data = {
            "bos_capex_kW": self.prob.get_val("landbosse.bos_capex_kW", units="USD/kW"),
            "total_capex": self.prob.get_val("landbosse.total_capex", units="MUSD"),
        }
        # validate data against pyrite file
        ard.utils.test_utils.pyrite_validator(
            test_data,
            fn_pyrite,
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )


class TestORBIT:

    def setup_method(self):

        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "ard"
            / "data"
            / "windIO-plant_turbine_IEA-22MW-284m-RWT.yaml"
        )

        # set up the modeling options
        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "turbine": ard.utils.io.load_yaml(filename_turbine),
                },
            },
            "layout": {
                "N_turbines": 25,
                "spacing_primary": 0.0,  # reset in test
                "spacing_secondary": 0.0,  # reset in test
                "angle_orientation": 0.0,  # reset in test
                "angle_skew": 0.0,  # reset in test
            },
            "costs": {
                "rated_power": 22000000.0,  # (W)
                "num_blades": 3,  # (-)
                "tower_length": 149.386,  # (m)
                "tower_mass": 1574044.87111,  # (tonne)
                "nacelle_mass": 849143.2357,  # (tonne)
                "blade_mass": 83308.31171,  # (tonne)
                "turbine_capex": 1397.17046735,  # (USD)
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
                # monopile_mass: 2097.21115974 # (t)
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
            "site_depth": 50.0,
            "offshore": True,
            "floating": True,
        }

        # create an OM model and problem
        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.orbit = self.model.add_subsystem(
            "orbit",
            wcost.ORBITGroup(
                modeling_options=self.modeling_options,
            ),
        )
        self.model.connect(  # effective primary spacing for BOS
            "spacing_effective_primary", "orbit.plant_turbine_spacing"
        )
        self.model.connect(  # effective secondary spacing for BOS
            "spacing_effective_secondary", "orbit.plant_row_spacing"
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_baseline_farm(self, subtests):

        self.prob.set_val("gridfarm.spacing_primary", 7.0)
        self.prob.set_val("gridfarm.spacing_secondary", 7.0)
        self.prob.set_val("gridfarm.angle_orientation", 0.0)
        self.prob.set_val("gridfarm.angle_skew", 0.0)

        self.prob.run_model()

        # use a file of pyrite-standard data to validate against
        fn_pyrite = Path(__file__).parent / "test_orbit_wrap_baseline_farm.npz"
        test_data = {
            "bos_capex": self.prob.get_val("orbit.bos_capex", units="USD"),
            "total_capex": self.prob.get_val("orbit.total_capex", units="MUSD"),
        }
        # validate data against pyrite file
        pyrite_data = ard.utils.test_utils.pyrite_validator(
            test_data,
            fn_pyrite,
            rtol_val=5e-3,
            load_only=True,
            # rewrite=True,  # uncomment to write new pyrite file
        )

        # Validate each key-value pair using subtests
        for key, value in test_data.items():
            with subtests.test(key=key):
                assert np.allclose(value, pyrite_data[key], rtol=5e-3), (
                    f"Mismatch for {key}: " f"expected {pyrite_data[key]}, got {value}"
                )


class TestPlantFinance:

    def setup_method(self):
        pass


class TestTurbineCapitalCosts:

    def setup_method(self):
        pass


class TestOperatingExpenses:

    def setup_method(self):
        pass
