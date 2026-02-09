import sys
from os import PathLike
from pathlib import Path

import numpy as np
import traceback


def pyrite_validator(
    data_for_validation: dict,
    filename_pyrite: PathLike,
    rewrite: bool = False,
    rtol_val: float = 1e-6,
    load_only: bool = False,
):
    """
    TO DO!!!
    """

    # get the basename if a suffix is provided
    filename_pyrite = Path(filename_pyrite).with_suffix("")

    if rewrite:
        # this helper function can write a file to hold pyrite-standard data

        # write out a npz file that holds the variables we want to be able to check
        np.savez(filename_pyrite, **data_for_validation)
        assert False
    else:
        # ... or it can check to make sure that an existing pyrite file matches the current data

        # load an existing pyrite-standard data file
        pyrite_data = np.load(filename_pyrite.with_suffix(".npz"))

        if load_only:
            return pyrite_data

        # for each of the variables in the pyrite-standard data file
        all_validation_matches = True
        for k, v in pyrite_data.items():
            # count how many of the values in the data match the equivalent validation data
            sum_isclose = np.sum(
                np.isclose(np.array(v), data_for_validation[k], rtol=rtol_val)
            )
            vd_size = np.array(data_for_validation[k], dtype=np.float64).size
            # assert all of the values match
            validation_matches = sum_isclose == vd_size

            if not validation_matches:
                print(f"for variable {k}:", file=sys.stderr)
                prefix = "\t"
                print(
                    prefix.join(
                        f"{sum_isclose} values match of {vd_size} total validation values"
                    ),
                    file=sys.stderr,
                )
                print(prefix.join(f"to a tolerance of {rtol_val:e}"), file=sys.stderr)
                print(f"pyrite data for {k}: {v}", file=sys.stderr)
                print(
                    f"computed data for {k}: {data_for_validation[k]}", file=sys.stderr
                )
                stack = traceback.format_stack(limit=3)
                print("".join(stack[-2]).replace("  File", "test:"), file=sys.stderr)
                print(file=sys.stderr)

            all_validation_matches &= validation_matches
        assert all_validation_matches, "Pyrite validation data must match."
