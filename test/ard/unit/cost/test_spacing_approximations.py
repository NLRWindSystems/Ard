import pytest
import openmdao.api as om

from ard.cost.approximate_turbine_spacing import SpacingApproximations


class TestSpacingApproximations:

    def setup_method(self):
        # Create the problem
        prob = om.Problem()

        # set modeling options
        modeling_options = {
            "windIO_plant": {"wind_farm": {"turbine": {"rotor_diameter": 10.0}}},
            "layout": {
                "N_turbines": 10,
            },
        }

        # Add the SpacingApproximations component with 10 turbines
        prob.model.add_subsystem(
            "spacing_calc",
            SpacingApproximations(modeling_options=modeling_options),
            promotes=["*"],
        )

        # Set up the problem
        prob.setup()

        # Set the input value
        prob.set_val("total_length_cables", 1000.0)  # Total cable length in meters

        # Run the model
        prob.run_model()

        self.prob = prob

    def test_turbine_spacing_calculation(self):
        """Test the turbine spacing calculation."""

        # Check the output value
        turbine_spacing = self.prob.get_val("primary_turbine_spacing_diameters")
        assert turbine_spacing == pytest.approx(10.0, abs=1e-12)

    def test_row_spacing_calculation(self):
        """Test the turbine spacing calculation."""

        # Check the output value
        row_spacing = self.prob.get_val("secondary_turbine_spacing_diameters")
        assert row_spacing == pytest.approx(10.0, abs=1e-12)

    def test_partial_derivatives(self):
        """Test the partial derivatives."""

        # Check the partial derivatives
        partials = self.prob.check_partials(out_stream=None)
        total_length_cables_partials = partials["spacing_calc"][
            ("primary_turbine_spacing_diameters", "total_length_cables")
        ]["J_fwd"]
        assert total_length_cables_partials == pytest.approx(0.01, abs=1e-12)
