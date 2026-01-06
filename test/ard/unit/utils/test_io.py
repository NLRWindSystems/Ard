from ard.utils.io import replace_key_value


class TestReplaceKeyValueNotNoneOnly:
    def setup_method(self):
        input_dictionary = {
            "a": 0,  # "a" at top level
            "b": [{"a": 0}, {}],  # "a" in a dictionary in a list of dictionaries
            "c": {"a": 0},  # "a" in a nested dictionary
        }
        self.output_dictionary = replace_key_value(
            input_dictionary, "a", 1, replace_none_only=False
        )

    def test_replace_key_value_top_level(self):
        assert self.output_dictionary["a"] == 1

    def test_replace_key_value_list_of_dictionaries(self):
        assert self.output_dictionary["b"][0]["a"] == 1

    def test_replace_key_value_nested_dictionary(self):
        assert self.output_dictionary["c"]["a"] == 1


class TestReplaceKeyValueNoneOnly:
    def setup_method(self):
        input_dictionary = {
            "a": 0,  # "a" at top level
            "b": [{"a": None}, {}],  # "a" in a dictionary in a list of dictionaries
            "c": {"a": 0},  # "a" in a nested dictionary
        }
        self.output_dictionary = replace_key_value(
            input_dictionary, "a", 1, replace_none_only=True
        )

    def test_replace_key_value_top_level(self):
        assert self.output_dictionary["a"] == 0

    def test_replace_key_value_list_of_dictionaries(self):
        assert self.output_dictionary["b"][0]["a"] == 1

    def test_replace_key_value_nested_dictionary(self):
        assert self.output_dictionary["c"]["a"] == 0
