from pathlib import Path
import pytest
import openmdao.api as om

import ard
import ard.utils.io
from ard.cost.wisdem_wrap import LandBOSSEWithSpacingApproximations


class TestLandBOSSEWithSpacingApproximations:
    def setup_method(self):

        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "ard"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )

        # set up the modeling options
        modeling_options = {
            "windIO_plant": {
                # "site": {
                #     "energy_resource": {
                #         "wind_resource": ard.utils.io.load_yaml(filename_windresource),
                #     },
                # },
                "wind_farm": {
                    "turbine": ard.utils.io.load_yaml(filename_turbine),
                    # "electrical_substations": [
                    #     {
                    #         "electrical_substation": {
                    #             "coordinates": {"x": [0.0], "y": [0.0]},
                    #         },
                    #     },
                    # ],
                },
            },
            "layout": {
                "N_turbines": 25,
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
        windIO_plant = modeling_options["windIO_plant"]

        # Create the problem
        prob = om.Problem()

        # Add the LandBOSSEWithSpacingApproximations group
        prob.model.add_subsystem(
            "landbosse_group",
            LandBOSSEWithSpacingApproximations(modeling_options=modeling_options),
            promotes=["*"],
        )

        # Set up the problem
        prob.setup()

        # Set the input value
        prob.set_val("total_length_cables", 1000.0)  # Total cable length in meters

        # Run the model
        prob.run_model()

        self.prob = prob
        self.modeling_options = modeling_options

    def test_primary_turbine_spacing(self):
        """Test the primary turbine spacing calculation."""
        # Check the output value
        primary_turbine_spacing = self.prob.get_val(
            "spacing_approximations.primary_turbine_spacing_diameters"
        )
        expected_spacing = 1000.0 / (
            self.modeling_options["layout"]["N_turbines"]
            * self.modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                "rotor_diameter"
            ]
        )
        assert primary_turbine_spacing == pytest.approx(expected_spacing, abs=1e-12)

    def test_secondary_turbine_spacing(self):
        """Test the primary turbine spacing calculation."""
        # Check the output value
        secondary_turbine_spacing = self.prob.get_val(
            "spacing_approximations.secondary_turbine_spacing_diameters"
        )
        expected_spacing = 1000.0 / (
            self.modeling_options["layout"]["N_turbines"]
            * self.modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                "rotor_diameter"
            ]
        )
        assert secondary_turbine_spacing == pytest.approx(expected_spacing, abs=1e-12)

    def test_internal_primary_turbine_spacing(self):
        """Test that the internal turbine spacing is passed correctly to LandBOSSE."""
        # Check that LandBOSSE receives the correct input
        internal_primary_turbine_spacing = self.prob.get_val(
            "internal_turbine_spacing_rotor_diameters"
        )
        primary_turbine_spacing = self.prob.get_val(
            "spacing_approximations.primary_turbine_spacing_diameters"
        )
        assert internal_primary_turbine_spacing == pytest.approx(
            primary_turbine_spacing, abs=1e-12
        )

    def test_internal_secondary_turbine_spacing(self):
        """Test that the internal turbine spacing is passed correctly to LandBOSSE."""
        # Check that LandBOSSE receives the correct input
        internal_secondary_turbine_spacing = self.prob.get_val(
            "internal_row_spacing_rotor_diameters"
        )
        secondary_turbine_spacing = self.prob.get_val(
            "spacing_approximations.secondary_turbine_spacing_diameters"
        )
        assert internal_secondary_turbine_spacing == pytest.approx(
            secondary_turbine_spacing, abs=1e-12
        )

    # def test_partial_derivatives(self):
    #     """Test the partial derivatives."""
    #     # Check the partial derivatives
    #     partials = self.prob.check_partials(out_stream=None, method="fd")
    #     total_length_cables_partials = partials["landbosse_group"][
    #         ("total_capex", "total_length_cables")
    #     ]["J_fwd"]
    #     expected_partial = 1.0 / (
    #         self.modeling_options["layout"]["N_turbines"]
    #         * self.options["modeling_options"]["windIO_plant"]["wind_farm"]["turbine"]["rotor_diameter"]
    #     )
    #     assert total_length_cables_partials == pytest.approx(expected_partial, abs=1E-12)
