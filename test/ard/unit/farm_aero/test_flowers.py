from pathlib import Path

import numpy as np
import pandas as pd

import openmdao.api as om

import flowers
import ard

import ard.farm_aero.flowers as farmaero_flowers
import ard.farm_aero.floris as farmaero_floris
import ard.utils.io


class TestFLOWERSPyrite:

    def setup_method(self):
        pass

    def test_pyrite(self, subtests):
        pass


class TestFLOWERSIndepMatch:

    def setup_method(self):

        self.path_csv_HKW = (
            Path(flowers.__file__).parents[1]
            / "examples"
            / "flowers"
            / "data"
            / "HKW_wind_rose.csv"
        )
        self.df_HKW = pd.read_csv(
            self.path_csv_HKW,
        )
        self.df_HKW["freq_val"] = np.maximum(
            self.df_HKW["freq_val"], 1.0e-6
        )  # prevent divide by zero

        N_turb = 256
        seed_val = int.from_bytes(b"0xAR")  # lock the seed for consistent results

        boundaries = [
            (-5000.0, 0.0),
            (0.0, 5000.0),
            (5000.0, 0.0),
            (0.0, -5000.0),
        ]  # a big diamond grid

        # generate a random layout
        xx, yy = self.layout = flowers.tools.random_layout(
            boundaries, N_turb, seed_val=seed_val
        )

        # grab the turbine_yaml and put it into a barebones windIO
        turbine_yaml = ard.utils.io.load_yaml(
            Path(__file__).parents[4]
            / "examples"
            / "ard"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        df_pivot = self.df_HKW.pivot(index="wd", columns="ws", values="freq_val")
        self.windIO_plant = {
            "wind_farm": {
                "turbine": turbine_yaml,
            },
            "site": {
                "energy_resource": {
                    "wind_resource": {
                        "wind_direction": df_pivot.index,
                        "wind_speed": df_pivot.columns,
                        "probability": {
                            "data": df_pivot.values,
                        },
                        "turbulence_intensity": {
                            "data": 0.06 * np.ones_like(df_pivot.values),
                        },
                    },
                },
            },
        }

        # roll up the data into a modeling_options
        self.modeling_options = {
            "windIO_plant": self.windIO_plant,
            "layout": {
                "N_turbines": N_turb,
            },
            "flowers": {
                "num_terms": 0,
                "k": 0.05,
            },
        }

        # use the floris turbine as an intermediary to cover off windIO variants
        self.floris_turbine = farmaero_floris.create_FLORIS_turbine_from_windIO(
            self.windIO_plant
        )

        # get the data that raw FLOWERS needs
        rho_density_air = self.floris_turbine["power_thrust_table"][
            "ref_air_density"
        ]  # kg/m^3
        area_rotor = np.pi / 4 * self.floris_turbine["rotor_diameter"] ** 2  # m^2
        V_table = np.array(self.floris_turbine["power_thrust_table"]["wind_speed"])
        P_table = 1.0e3 * np.array(self.floris_turbine["power_thrust_table"]["power"])
        CT_table = np.array(
            self.floris_turbine["power_thrust_table"]["thrust_coefficient"]
        )
        CP_table = np.where(
            V_table == 0.0,
            0.0,
            P_table / (0.5 * rho_density_air * area_rotor * V_table**3),
        )

        # extract the turbine from the windIO for FLOWERS
        turbine_type = {
            "D": self.windIO_plant["wind_farm"]["turbine"]["rotor_diameter"],
            "U": self.windIO_plant["wind_farm"]["turbine"]["performance"].get(
                "cutout_wind_speed", 25.0
            ),
            "ct": CT_table,
            "u_ct": V_table,
            "cp": CP_table,
            "u_cp": V_table,
        }

        # create a flowers model
        self.fm = flowers.FlowersModel(
            wind_rose=self.df_HKW,
            layout_x=xx,
            layout_y=yy,
            num_terms=self.modeling_options["flowers"]["num_terms"],
            k=self.modeling_options["flowers"]["k"],
            turbine=turbine_type,
        )

        # create a problem instance
        prob = self.prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            "flowers",
            farmaero_flowers.FLOWERSAEP(
                modeling_options=self.modeling_options,
            ),
        )
        prob.setup()
        prob.set_val("flowers.x_turbines", xx, units="m")
        prob.set_val("flowers.y_turbines", yy, units="m")

    def test_instantiation(self, subtests):

        self.prob.run_model()  # flowers_model not created until compute

        # extract the flowers models
        fm_ard = self.prob.model.flowers.flowers_model
        fm = self.fm

        # make sure parameters match
        with subtests.test("parameter k isclose"):
            assert np.isclose(fm_ard.k, fm.k)
        with subtests.test("parameter num_modes isclose"):
            # should have the same number of modes
            assert np.isclose(fm_ard.get_num_modes(), fm.get_num_modes())

        # make sure the wind rose matches
        with subtests.test("windrose wd allclose"):
            assert np.allclose(fm.wind_rose["wd"], fm_ard.wind_rose["wd"])
        with subtests.test("windrose ws allclose"):
            assert np.allclose(fm.wind_rose["ws"], fm_ard.wind_rose["ws"])
        with subtests.test("windrose freq_val allclose"):
            assert np.allclose(
                fm_ard.wind_rose["freq_val"],
                fm.wind_rose["freq_val"],
                atol=1.0e-5,  # freq val gets truncated for some reason
            )

        # make sure the layouts match
        with subtests.test("layout_x allclose"):
            assert np.allclose(fm_ard.layout_x, fm.layout_x)
        with subtests.test("layout_y allclose"):
            assert np.allclose(fm_ard.layout_y, fm.layout_y)

    def test_standalone_match(self):

        # run the standalone flowers as a reference value
        AEP_standalone_flowers = self.fm.calculate_aep() / 1.0e9

        # run Ard flowers to make sure it matches
        self.prob.run_model()
        AEP_ard_flowers = self.prob.get_val("flowers.AEP_farm", units="GW*h")[0]

        # make sure standalone FLOWERS and Ard FLOWERS match
        assert np.isclose(AEP_ard_flowers, AEP_standalone_flowers, rtol=0.005)
        # loose tolerance for roundoff in wind rose... not sure where it comes from
