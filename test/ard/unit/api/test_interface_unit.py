import pytest

from pathlib import Path
from ard.api import set_up_ard_model

import jsonschema.exceptions


class TestSetUpArdModel:
    def setup_method(self):
        pass

    def test_invalid_default_system(self):
        input_dict = {
            "system": "test",
            "modeling_options": {},
            "analysis_options": {},
        }

        with pytest.raises(
            ValueError,
            match=f"invalid default system 'test' specified. Must be one of "
            "\\['onshore', 'onshore_batch', 'onshore_no_cable_design', "
            "'offshore_monopile', 'offshore_monopile_no_cable_design', "
            "'offshore_floating', 'offshore_floating_no_cable_design'\\]",
        ):
            set_up_ard_model(input_dict)


class TestSetUpArdModelInvalidWindIO:
    def setup_method(self):

        self.input_dict_path = str(
            Path(__file__).parent.absolute()
            / "inputs_onshore"
            / "ard_system_bad_windio.yaml"
        )

    def test_windIO_validation_error(self):

        with pytest.raises(
            jsonschema.exceptions.ValidationError, match="'y' is a required property"
        ):
            self.prob = set_up_ard_model(input_dict=self.input_dict_path)
