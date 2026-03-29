from pathlib import Path

import numpy as np

from . import test_rotational_workbench


class TestFLOWERSRotationalConsistency:
    """
    these are tests to make sure that FLOWERS matches data we've compared
    against FLORIS to make sure that the behavior of flowers matches what we
    have previously calculated, particularly on a farm with no rotational
    symmetries that could hide simple rotational mistakes
    """

    def setup_method(self):
        path_rotational_file = Path(__file__).parent / "rotational_consistency.npz"
        self.reference_pyrite_data = np.load(path_rotational_file, allow_pickle=True)

    def test_wind_direction_pyrite_match(self, subtests):
        """test for a pyrite match with the orientation experiment"""

        # extract the saved off data
        flowers_turbine = self.reference_pyrite_data[
            "flowers_turbine"
        ].tolist()  # needed to unpack
        wd_vec = self.reference_pyrite_data["wd_vec"]
        AEP_FLOWERS_vec_ref = self.reference_pyrite_data["AEP_FLOWERS_e1_vec"]

        # run FLOWERS using the shared toolset
        AEP_FLOWERS_vec = np.zeros_like(wd_vec)
        for idx, wd_val in enumerate(wd_vec):
            AEP_FLOWERS_vec[idx] = test_rotational_workbench.run_FLOWERS(
                flowers_turbine, wd_val=wd_val
            )

        # assert that the reference matches what we calculated
        assert np.allclose(AEP_FLOWERS_vec, AEP_FLOWERS_vec_ref)

    def test_orientation_pyrite_match(self, subtests):
        """test for a pyrite match with the orientation experiment"""

        # extract the saved off data
        flowers_turbine = self.reference_pyrite_data[
            "flowers_turbine"
        ].tolist()  # needed to unpack
        orientation_vec = self.reference_pyrite_data["orientation_vec"]
        AEP_FLOWERS_vec_ref = self.reference_pyrite_data["AEP_FLOWERS_e2_vec"]

        # run FLOWERS using the shared toolset
        AEP_FLOWERS_vec = np.zeros_like(orientation_vec)
        for idx, orientation_val in enumerate(orientation_vec):
            AEP_FLOWERS_vec[idx] = test_rotational_workbench.run_FLOWERS(
                flowers_turbine, orientation=orientation_val
            )

        # assert that the reference matches what we calculated
        assert np.allclose(AEP_FLOWERS_vec, AEP_FLOWERS_vec_ref)

    def test_FLORIS_wind_direction_reference_agreement(self, subtests):

        AEP_FLORIS_e1_vec = self.reference_pyrite_data["AEP_FLORIS_e1_vec"]
        AEP_FLOWERS_e1_vec = self.reference_pyrite_data["AEP_FLOWERS_e1_vec"]

        # compute the R squared value and assert that it improves from our baseline
        Rsquared = np.corrcoef(AEP_FLOWERS_e1_vec, AEP_FLORIS_e1_vec)[0, 1] ** 2
        assert Rsquared >= 0.95 * 0.935  # relax base value from reference data

    def test_FLORIS_orientation_reference_agreement(self, subtests):

        AEP_FLORIS_e2_vec = self.reference_pyrite_data["AEP_FLORIS_e2_vec"]
        AEP_FLOWERS_e2_vec = self.reference_pyrite_data["AEP_FLOWERS_e2_vec"]

        # compute the R squared value and assert that it improves from our baseline
        Rsquared = np.corrcoef(AEP_FLOWERS_e2_vec, AEP_FLORIS_e2_vec)[0, 1] ** 2
        assert Rsquared >= 0.95 * 0.874  # relax base value from reference data
