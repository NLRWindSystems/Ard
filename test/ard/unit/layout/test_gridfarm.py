import numpy as np
import openmdao.api as om

import pytest

import ard.layout.gridfarm as gridfarm


class TestGridFarm:

    def setup_method(self):

        self.N_turbines = 25
        self.D_rotor = 130.0

        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    },
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
                "spacing_primary": 0.0,  # reset in test_setup
                "spacing_secondary": 0.0,  # reset in test_setup
                "angle_orientation": 0.0,  # reset in test_setup
                "angle_skew": 0.0,  # reset in test_setup
            },
        }

        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling_options has what we need for the layout
        assert "modeling_options" in [k for k, _ in self.gf.options.items()]

        assert "layout" in self.gf.options["modeling_options"].keys()
        assert "N_turbines" in self.gf.options["modeling_options"]["layout"].keys()

        assert "wind_farm" in self.modeling_options["windIO_plant"].keys()
        assert "turbine" in self.modeling_options["windIO_plant"]["wind_farm"].keys()
        assert (
            "rotor_diameter"
            in self.modeling_options["windIO_plant"]["wind_farm"]["turbine"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.gf.list_inputs()]
            for var_to_check in [
                "spacing_primary",
                "spacing_secondary",
                "angle_orientation",
                "angle_skew",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.gf.list_outputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "spacing_effective_primary",
                "spacing_effective_secondary",
            ]:
                assert var_to_check in output_list

    def test_compute_squarefarm(self):

        for spacing1, spacing2 in [(3.0, 5.0), (7.0, 4.0)]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", 0.0)
            self.prob.set_val("gridfarm.angle_skew", 0.0)

            self.prob.run_model()

            x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
            y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
            Y, X = np.meshgrid(y_turbines, x_turbines)

            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.x_turbines"), X.flatten())
            )
            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.y_turbines"), Y.flatten())
            )

    def test_compute_rotatedfarm(self):

        angle_skew = 0.0
        for spacing1, spacing2, angle_orientation in [
            (3.0, 5.0, 7.5),
            (7.0, 4.0, -4.5),
        ]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
            self.prob.set_val("gridfarm.angle_skew", angle_skew)

            self.prob.run_model()

            x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
            y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
            Y, X = np.meshgrid(y_turbines, x_turbines)
            Xr = (
                np.cos(np.radians(angle_orientation)) * X
                + np.sin(np.radians(angle_orientation)) * Y
            )
            Yr = (
                -np.sin(np.radians(angle_orientation)) * X
                + np.cos(np.radians(angle_orientation)) * Y
            )

            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.x_turbines"), Xr.flatten())
            )
            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.y_turbines"), Yr.flatten())
            )

    def test_compute_skewedfarm(self):

        angle_orientation = 0.0
        for spacing1, spacing2, angle_skew in [
            (3.0, 5.0, 12.5),
            (7.0, 4.0, 27.5),
        ]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
            self.prob.set_val("gridfarm.angle_skew", angle_skew)

            self.prob.run_model()

            x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
            y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
            Y, X = np.meshgrid(y_turbines, x_turbines)
            Xs = X
            Ys = X * np.tan(-np.radians(angle_skew)) + Y
            Xr, Yr = Xs, Ys

            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.x_turbines"), Xr.flatten())
            )
            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.y_turbines"), Yr.flatten())
            )

    def test_compute_rotatedskewedfarm(self):

        for spacing1, spacing2, angle_orientation, angle_skew in [
            (3.0, 5.0, 7.5, 12.5),
            (7.0, 4.0, -4.5, 27.5),
        ]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
            self.prob.set_val("gridfarm.angle_skew", angle_skew)

            self.prob.run_model()

            x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
            y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
            Y, X = np.meshgrid(y_turbines, x_turbines)
            Xs = X
            Ys = -np.tan(np.radians(angle_skew)) * X + Y
            Xr = (
                np.cos(np.radians(angle_orientation)) * Xs
                + np.sin(np.radians(angle_orientation)) * Ys
            )
            Yr = (
                -np.sin(np.radians(angle_orientation)) * Xs
                + np.cos(np.radians(angle_orientation)) * Ys
            )

            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.x_turbines"), Xr.flatten())
            )
            assert np.all(
                np.isclose(self.prob.get_val("gridfarm.y_turbines"), Yr.flatten())
            )


class TestGridFarmLanduse:

    def setup_method(self):

        self.N_turbines = 25
        self.D_rotor = 130.0

        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    },
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
                "spacing_primary": 0.0,  # reset in test_setup
                "spacing_secondary": 0.0,  # reset in test_setup
                "angle_orientation": 0.0,  # reset in test_setup
                "angle_skew": 0.0,  # reset in test_setup
            },
        }

        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.lu = self.model.add_subsystem(
            "gflu",
            gridfarm.GridFarmLanduse(
                modeling_options=self.modeling_options,
            ),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling_options has what we need for the layout
        assert "modeling_options" in [k for k, _ in self.lu.options.items()]

        assert "layout" in self.lu.options["modeling_options"].keys()
        assert "N_turbines" in self.lu.options["modeling_options"]["layout"].keys()

        assert "wind_farm" in self.modeling_options["windIO_plant"].keys()
        assert "turbine" in self.modeling_options["windIO_plant"]["wind_farm"].keys()
        assert (
            "rotor_diameter"
            in self.modeling_options["windIO_plant"]["wind_farm"]["turbine"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.lu.list_inputs()]
            for var_to_check in [
                "spacing_primary",
                "spacing_secondary",
                "angle_orientation",
                "angle_skew",
                "distance_layback_diameters",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.lu.list_outputs()]
            for var_to_check in [
                "area_tight",
                "area_aligned_parcel",
                "area_compass_parcel",
            ]:
                assert var_to_check in output_list

    def test_compute_squarefarm(self):

        for spacing1, spacing2 in [(3.0, 5.0), (7.0, 4.0)]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", 0.0)
            self.prob.set_val("gridfarm.angle_skew", 0.0)

            self.prob.run_model()

            x_min = np.min(self.prob.get_val("gridfarm.x_turbines", units="km"))
            x_max = np.max(self.prob.get_val("gridfarm.x_turbines", units="km"))
            y_min = np.min(self.prob.get_val("gridfarm.y_turbines", units="km"))
            y_max = np.max(self.prob.get_val("gridfarm.y_turbines", units="km"))

            A_ref = (
                (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing1
                * spacing2
                * self.D_rotor**2
                / (1000.0) ** 2
            )  # tight area
            A_skew = (
                A_ref
                + (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing2**2
                * self.D_rotor**2
                * np.abs(np.tan(np.radians(0.0)))
                / (1000.0) ** 2
            )  # skewed area on to primary farm dim
            A_square = np.abs(
                (x_max - x_min) * (y_max - y_min)
            )  # compass-aligned square

            # for a square, unskewed, compass-aligned farm all areas are equal
            assert np.isclose(self.prob.get_val("gflu.area_tight"), A_ref)
            assert np.isclose(self.prob.get_val("gflu.area_aligned_parcel"), A_skew)
            assert np.isclose(self.prob.get_val("gflu.area_compass_parcel"), A_square)

    def test_compute_rotatedfarm(self):

        angle_skew = 0.0
        for spacing1, spacing2, angle_orientation in [
            (3.0, 5.0, 7.5),
            (7.0, 4.0, -4.5),
        ]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
            self.prob.set_val("gridfarm.angle_skew", angle_skew)

            self.prob.run_model()

            x_min = np.min(self.prob.get_val("gridfarm.x_turbines", units="km"))
            x_max = np.max(self.prob.get_val("gridfarm.x_turbines", units="km"))
            y_min = np.min(self.prob.get_val("gridfarm.y_turbines", units="km"))
            y_max = np.max(self.prob.get_val("gridfarm.y_turbines", units="km"))

            A_ref = (
                (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing1
                * spacing2
                * self.D_rotor**2
                / (1000.0) ** 2
            )  # tight area
            A_skew = (
                A_ref
                + (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing2**2
                * self.D_rotor**2
                * np.abs(np.tan(np.radians(angle_skew)))
                / (1000.0) ** 2
            )  # skewed area on to primary farm dim
            A_square = np.abs(
                (x_max - x_min) * (y_max - y_min)
            )  # compass-aligned square

            # for a rotated, unskewed, compass-aligned farm first 2 areas are equal
            assert np.isclose(self.prob.get_val("gflu.area_tight"), A_ref)
            assert np.isclose(self.prob.get_val("gflu.area_aligned_parcel"), A_skew)
            assert np.isclose(self.prob.get_val("gflu.area_compass_parcel"), A_square)

    def test_compute_skewedfarm(self):

        angle_orientation = 0.0
        for spacing1, spacing2, angle_skew in [(4.0, 5.5, 14.5), (3.0, 6.5, -5.25)]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
            self.prob.set_val("gridfarm.angle_skew", angle_skew)

            self.prob.run_model()

            x_min = np.min(self.prob.get_val("gridfarm.x_turbines", units="km"))
            x_max = np.max(self.prob.get_val("gridfarm.x_turbines", units="km"))
            y_min = np.min(self.prob.get_val("gridfarm.y_turbines", units="km"))
            y_max = np.max(self.prob.get_val("gridfarm.y_turbines", units="km"))

            A_ref = (
                (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing1
                * spacing2
                * self.D_rotor**2
                / (1000.0) ** 2
            )  # tight area
            A_skew = (
                A_ref
                + (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing2**2
                * self.D_rotor**2
                * np.abs(np.tan(np.radians(angle_skew)))
                / (1000.0) ** 2
            )  # skewed area on to primary farm dim
            A_square = np.abs(
                (x_max - x_min) * (y_max - y_min)
            )  # compass-aligned square

            # for a skewed, compass-aligned farm second and third areas are equal
            assert np.isclose(self.prob.get_val("gflu.area_tight"), A_ref)
            assert np.isclose(self.prob.get_val("gflu.area_aligned_parcel"), A_skew)
            assert np.isclose(self.prob.get_val("gflu.area_compass_parcel"), A_square)

    def test_compute_rotatedskewedfarm(self):

        for spacing1, spacing2, angle_orientation, angle_skew in [
            (3.0, 5.0, 7.5, 12.5),
            (7.0, 4.0, -4.5, 27.5),
        ]:

            self.prob.set_val("gridfarm.spacing_primary", spacing1)
            self.prob.set_val("gridfarm.spacing_secondary", spacing2)
            self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
            self.prob.set_val("gridfarm.angle_skew", angle_skew)

            self.prob.run_model()

            x_min = np.min(self.prob.get_val("gridfarm.x_turbines", units="km"))
            x_max = np.max(self.prob.get_val("gridfarm.x_turbines", units="km"))
            y_min = np.min(self.prob.get_val("gridfarm.y_turbines", units="km"))
            y_max = np.max(self.prob.get_val("gridfarm.y_turbines", units="km"))

            A_ref = (
                (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing1
                * spacing2
                * self.D_rotor**2
                / (1000.0) ** 2
            )  # tight area
            A_skew = (
                A_ref
                + (np.sqrt(self.N_turbines) - 1) ** 2
                * spacing2**2
                * self.D_rotor**2
                * np.abs(np.tan(np.radians(angle_skew)))
                / (1000.0) ** 2
            )  # skewed area on to primary farm dim
            A_square = np.abs(
                (x_max - x_min) * (y_max - y_min)
            )  # compass-aligned square

            # for a skewed, compass-aligned farm second and third areas are equal
            assert np.isclose(self.prob.get_val("gflu.area_tight"), A_ref)
            assert np.isclose(self.prob.get_val("gflu.area_aligned_parcel"), A_skew)
            assert np.isclose(self.prob.get_val("gflu.area_compass_parcel"), A_square)
