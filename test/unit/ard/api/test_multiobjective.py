from pathlib import Path

import openmdao.api as om

from ard.utils.io import load_yaml
from ard.api import set_up_ard_model

import pytest


class TestMultiobjectiveSetUp:
    def setup_method(self):

        # create the simplest system that will compile
        input_dict = self.input_dict = load_yaml(
            Path(__file__).parent / "inputs_onshore" / "ard_system_multiobjective.yaml"
        )

        # create an ard model
        self.da_plough = set_up_ard_model(
            input_dict=input_dict,
        )

    def teardown_method(self):

        # cleanup the ard model
        self.da_plough.cleanup()
        # necessary due to final_setup() call below?

    def test_response_variables(self, subtests):

        # preemptively run the final setup
        self.da_plough.final_setup()

        # extract the response vars
        response_vars = self.da_plough.list_driver_vars(out_stream=None)

        # the expected responses
        DVs_expected = {
            "spacing_primary",
            "spacing_secondary",
            "angle_orientation",
            "angle_skew",
        }
        constrs_expected = {
            "boundary_distances",
        }
        objs_expected = {
            "AEP_farm",
            "lug.landuse.area_tight",
        }

        # use set equivalence to make sure the OM problem matches expectations
        with subtests.test("design vars"):
            assert (
                set([v[0] for v in response_vars["design_vars"]]) == DVs_expected
            ), "design vars must match"
        with subtests.test("constraints"):
            assert (
                set([v[0] for v in response_vars["constraints"]]) == constrs_expected
            ), "constraints must match"
        with subtests.test("objectives"):
            assert (
                set([v[0] for v in response_vars["objectives"]]) == objs_expected
            ), "objectives must match"

    def test_raise_scipy_MOO_error(self):

        self.input_dict["analysis_options"]["driver"]["name"] = "ScipyOptimizeDriver"
        self.input_dict["analysis_options"]["driver"]["options"] = {
            "optimizer": "COBYLA",
            "opt_settings": {
                "rhobeg": 2.0,
                "maxiter": 50,
            },
        }

        # re-create an ard model
        self.da_plough = set_up_ard_model(
            input_dict=self.input_dict,
        )

        # make sure the driver runs and gets the scipy error
        with pytest.raises(
            RuntimeError,
            match="ScipyOptimizeDriver currently does not support multiple objectives.",
        ):

            # attempt to run the driver
            self.da_plough.run_driver()
