import numpy as np
import openmdao.api as om

import pytest

import ard.collection.templates as templates


class TestCollectionTemplate:

    def setup_method(self):
        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "electrical_substations": [
                        {
                            "electrical_substation": {
                                "coordinates": {
                                    "x": [0.0, 0.0],
                                    "y": [0.0, 0.0],
                                },
                            },
                        },
                    ],
                },
            },
            "layout": {
                "N_turbines": 4,
                "N_substations": 2,
                "x_turbines": np.zeros(4),
                "y_turbines": np.zeros(4),
            },
        }

        self.model = om.Group()
        self.coll_temp = self.model.add_subsystem(
            "coll_temp",
            templates.CollectionTemplate(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        """
        make sure the modeling_options has what we need for farmaero
        """

        assert "modeling_options" in [k for k, _ in self.coll_temp.options.items()]

        assert "windIO_plant" in self.coll_temp.options["modeling_options"].keys()
        assert (
            "wind_farm"
            in self.coll_temp.options["modeling_options"]["windIO_plant"].keys()
        )
        assert (
            "N_turbines" in self.coll_temp.options["modeling_options"]["layout"].keys()
        )
        assert (
            "N_substations"
            in self.coll_temp.options["modeling_options"]["layout"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the inputs in the component match what we planned
            input_list = [k for k, v in self.coll_temp.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.coll_temp.list_outputs()]
            for var_to_check in [
                "length_cables",
                "load_cables",
                "total_length_cables",
                "max_load_cables",
            ]:
                assert var_to_check in output_list

            # make sure that the outputs in the component match what we planned
            discrete_output_list = [
                k for k, v in self.coll_temp._discrete_outputs.items()
            ]
            for var_to_check in [
                "terse_links",
            ]:
                assert var_to_check in discrete_output_list

    def test_compute(self):

        # make sure that an attempt to compute on the un-specialized class fails
        with pytest.raises(NotImplementedError):
            x_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            y_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            x_substations = np.array([-1, 1]) * [3.5 * 130.0]
            y_substations = np.array([-1, 1]) * [3.5 * 130.0]
            self.prob.set_val("x_turbines", x_turbines)
            self.prob.set_val("y_turbines", y_turbines)
            self.prob.set_val("x_substations", x_substations)
            self.prob.set_val("y_substations", y_substations)
            self.prob.run_model()
