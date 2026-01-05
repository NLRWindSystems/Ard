from pathlib import Path

import numpy as np

from wisdem.optimization_drivers.nsga2_driver import NSGA2Driver

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


class TestNSGA2:
    def setup_method(self):

        # create the simplest system that will compile
        input_dict = self.input_dict = load_yaml(
            Path(__file__).parent / "inputs_onshore" / "ard_system_NSGA2.yaml"
        )

        # create an ard model
        self.da_plough = set_up_ard_model(
            input_dict=input_dict,
        )

    def teardown_method(self):

        # cleanup the ard model
        self.da_plough.cleanup()
        # necessary due to final_setup() call below?

    def test_instantiation(self, subtests):

        # make sure the driver is the right type
        with subtests.test("driver type"):
            assert type(self.da_plough.driver) == NSGA2Driver

        # make sure the default parameters are in the driver
        for opt_name, opt_val, comparison_fun in [
            ("run_parallel", False, np.equal),
            ("procs_per_model", 1, np.equal),
            ("penalty_parameter", 0.0, np.isclose),
            ("penalty_exponent", 1.0, np.isclose),
            ("compute_pareto", True, np.equal),
        ]:
            with subtests.test(f"driver default {opt_name}"):
                assert comparison_fun(self.da_plough.driver.options[opt_name], opt_val)

        # make sure the parameters we set in the ard yaml are set
        with subtests.test("driver setting max_gen"):
            assert (
                self.da_plough.driver.options["max_gen"]
                == self.input_dict["analysis_options"]["driver"]["options"]["max_gen"]
            )
        with subtests.test("driver setting pop_size"):
            assert (
                self.da_plough.driver.options["pop_size"]
                == self.input_dict["analysis_options"]["driver"]["options"]["pop_size"]
            )
        with subtests.test("driver setting Pc"):
            assert (
                self.da_plough.driver.options["Pc"]
                == self.input_dict["analysis_options"]["driver"]["options"]["Pc"]
            )
        with subtests.test("driver setting eta_c"):
            assert (
                self.da_plough.driver.options["eta_c"]
                == self.input_dict["analysis_options"]["driver"]["options"]["eta_c"]
            )
        with subtests.test("driver setting Pm"):
            assert (
                self.da_plough.driver.options["Pm"]
                == self.input_dict["analysis_options"]["driver"]["options"]["Pm"]
            )
        with subtests.test("driver setting eta_m"):
            assert (
                self.da_plough.driver.options["eta_m"]
                == self.input_dict["analysis_options"]["driver"]["options"]["eta_m"]
            )

    def test_driver_run(self, subtests):

        # make sure the driver runs to completion
        self.da_plough.run_driver()
