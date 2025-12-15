from pathlib import Path

import numpy as np

import windIO

import ard
import ard.utils.test_utils
from ard.utils.io import load_yaml

import ard.api.interface as glue


class TestLCOE_OFB_stack:

    def setup_method(self):

        # load the Ard system input
        path_ard_system = (
            Path(__file__).parent / "inputs_offshore_monopile" / "ard_system.yaml"
        )
        input_dict = load_yaml(path_ard_system)
        input_dict["modeling_options"]["case_name"] = (
            input_dict["modeling_options"]["case_name"] + "_serial"
        )

        # get, validate, and load the windIO dict
        windIOdict = input_dict["modeling_options"]["windIO_plant"]
        windIO.validate(windIOdict, schema_type="plant/wind_energy_system")

        # build an Ard model using the setup
        self.prob = glue.set_up_ard_model(
            input_dict=input_dict, root_data_path="inputs_onshore"
        )

    def teardown_method(self):

        # cleanup the ard model
        self.prob.cleanup()
        # necessary due to something about windows???

    def test_model(self, subtests):

        # set up the working/design variables
        self.prob.set_val("spacing_primary", 7.0)
        self.prob.set_val("spacing_secondary", 7.0)
        self.prob.set_val("angle_orientation", 0.0)
        self.prob.set_val("angle_skew", 0.0)

        # run the model
        self.prob.run_model()

        # collapse the test result data
        test_data = {
            "AEP_val": float(self.prob.get_val("AEP_farm", units="GW*h")[0]),
            "CapEx_val": float(self.prob.get_val("tcc.tcc", units="MUSD")[0]),
            "BOS_val": float(
                self.prob.get_val("orbit.installation_capex", units="MUSD")[0]
            ),
            "OpEx_val": float(self.prob.get_val("opex.opex", units="MUSD/yr")[0]),
            "LCOE_val": float(self.prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        }

        # check the data against a pyrite file
        pyrite_data = ard.utils.test_utils.pyrite_validator(
            test_data,
            Path(ard.__file__).parents[1]
            / "test"
            / "system"
            / "ard"
            / "api"
            / "test_LCOE_OFB_stack_pyrite.npz",
            # rewrite=True,  # uncomment to write new pyrite file
            rtol_val=5e-3,
            load_only=True,
        )

        # Validate each key-value pair using subtests
        for key, value in test_data.items():
            with subtests.test(key=key):
                assert np.isclose(value, pyrite_data[key], rtol=5e-3), (
                    f"Mismatch for {key}: " f"expected {pyrite_data[key]}, got {value}"
                )
