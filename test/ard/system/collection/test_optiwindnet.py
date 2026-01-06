from pathlib import Path
import pytest

# import platform


import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import ard
import ard.utils.test_utils
import ard.utils.io
import ard.wind_query as wq
import ard.layout.sunflower as sunflower
import ard.farm_aero.floris as farmaero_floris
import ard.collection.optiwindnet_wrap as inter


@pytest.mark.usefixtures("subtests")
class TestoptiwindnetLayout:

    def setup_method(self):

        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        filename_windresource = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_wind-resource_wrg-example.yaml"
        )

        # set up the modeling options
        self.modeling_options = {
            "windIO_plant": {
                "site": {
                    "energy_resource": {
                        "wind_resource": ard.utils.io.load_yaml(filename_windresource),
                    },
                },
                "wind_farm": {
                    "turbine": ard.utils.io.load_yaml(filename_turbine),
                    "electrical_substations": [
                        {
                            "electrical_substation": {
                                "coordinates": {"x": [0.0], "y": [0.0]},
                            },
                        },
                    ],
                },
            },
            "layout": {
                "N_turbines": 25,
                "N_substations": 1,
                "x_turbines": np.zeros(25),
                "y_turbines": np.zeros(25),
            },
            "aero": {
                "return_turbine_output": True,
            },
            "floris": {
                "peak_shaving_fraction": 0.4,
                "peak_shaving_TI_threshold": 0.0,
            },
            "offshore": False,
            "collection": {
                "max_turbines_per_string": 8,
                "model_options": dict(
                    topology="branched",
                    feeder_route="segmented",
                    feeder_limit="unlimited",
                ),
                "solver_name": "highs",
                "solver_options": dict(
                    time_limit=60,
                    mip_gap=0.005,  # TODO ???
                ),
            },
        }

        # create the OpenMDAO model
        self.model = om.Group()
        self.model.add_subsystem(  # layout component
            "layout",
            sunflower.SunflowerFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.model.add_subsystem(  # landuse component
            "landuse",
            sunflower.SunflowerFarmLanduse(modeling_options=self.modeling_options),
            promotes_inputs=["*"],
        )
        self.model.add_subsystem(  # FLORIS AEP component
            "aepFLORIS",
            farmaero_floris.FLORISAEP(
                modeling_options=self.modeling_options,
                case_title="letsgo",
                data_path="",
            ),
            # promotes=["AEP_farm"],
            promotes=["x_turbines", "y_turbines", "AEP_farm"],
        )
        self.model.add_subsystem(
            "collection",
            inter.OptiwindnetCollection(modeling_options=self.modeling_options),
            promotes=["x_turbines", "y_turbines"],
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_model(self, subtests):

        # set up the working/design variables
        self.prob.set_val("spacing_target", 7.0)
        self.prob.set_val("collection.x_substations", [0.0])
        self.prob.set_val("collection.y_substations", [0.0])

        # run the model
        self.prob.run_model()

        # approximated circle area should be close to the sunflower area
        area_circle = (
            np.pi
            / 4
            * np.ptp(self.prob.get_val("x_turbines", units="km"))
            * np.ptp(self.prob.get_val("y_turbines", units="km"))
        )

        with subtests.test("land use area"):
            assert np.isclose(
                self.prob.get_val("landuse.area_tight"), area_circle, rtol=0.1
            )

        # collect optiwindnet data to validate
        validation_data = {
            "terse_links": self.prob.get_val("collection.terse_links"),
            "length_cables": self.prob.get_val("collection.length_cables"),
            "load_cables": self.prob.get_val("collection.load_cables"),
            "total_length_cables": self.prob.get_val("collection.total_length_cables"),
            "max_load_cables": self.prob.get_val("collection.max_load_cables"),
        }

        with subtests.test("pyrite validator"):
            ard.utils.test_utils.pyrite_validator(
                validation_data,
                Path(__file__).parent / "test_optiwindnet_pyrite.npz",
                rtol_val=5e-3,
                #  rewrite=True,  # uncomment to write new pyrite file
            )

        # os_name = platform.system()

        # if os_name == 'Linux':
        #     # Run Linux specific tests
        #     # validate data against pyrite file

        # elif os_name == 'Darwin':
        #     # Run macos specific tests
        #     # validate data against pyrite file
        #     with subtests.test("pyrite validator"):
        #         ard.test_utils.pyrite_validator(
        #             validation_data,
        #             Path(__file__).parent / "test_optiwindnet_pyrite_macos.npz",
        #             rtol_val=5e-3,
        #             # rewrite=True,  # uncomment to write new pyrite file
        #         )
        # elif os_name == "Windows":
        #     # Run Windows specific tests
        #     # validate data against pyrite file
        #     with subtests.test("pyrite validator"):
        #         ard.test_utils.pyrite_validator(
        #             validation_data,
        #             Path(__file__).parent / "test_optiwindnet_pyrite_macos.npz",
        #             rtol_val=5e-3,
        #             # rewrite=True,  # uncomment to write new pyrite file
        #         )
        # else:
        #     raise(ValueError("Invalid OS for pyrite validation test"))
