import numpy as np
import openmdao.api as om

import pytest

import ard.layout.templates as layout_templates


class TestLayoutTemplate:

    def setup_method(self):

        self.modeling_options = {
            "windIO_plant": {},
            "layout": {
                "N_turbines": 4,
            },
        }

        self.model = om.Group()
        self.lt = self.model.add_subsystem(
            "layout",
            layout_templates.LayoutTemplate(
                modeling_options=self.modeling_options,
            ),
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling options has the things we need for the layout
        assert "modeling_options" in [k for k, _ in self.lt.options.items()]
        assert "layout" in self.lt.options["modeling_options"].keys()
        assert "N_turbines" in self.lt.options["modeling_options"]["layout"].keys()

        # make sure that the outputs in the component match what we planned
        output_list = [k for k, v in self.lt.list_outputs()]
        for var_to_check in [
            "x_turbines",
            "y_turbines",
            "spacing_effective_primary",
            "spacing_effective_secondary",
        ]:
            assert var_to_check in output_list

    def test_compute(self):

        # make sure that an attempt to compute on the un-specialized class fails
        with pytest.raises(NotImplementedError):
            self.prob.run_model()


class TestLanduseTemplate:

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
            },
        }

        self.model = om.Group()
        self.lu = self.model.add_subsystem(
            "landuse",
            layout_templates.LanduseTemplate(
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

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.lu.list_inputs()]
            for var_to_check in [
                "distance_layback_diameters",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.lu.list_outputs()]
            for var_to_check in [
                "area_tight",
            ]:
                assert var_to_check in output_list

    def test_compute(self):

        # make sure that an attempt to compute on the un-specialized class fails
        with pytest.raises(NotImplementedError):
            self.prob.run_model()
