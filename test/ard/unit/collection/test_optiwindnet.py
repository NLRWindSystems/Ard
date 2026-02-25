import copy
from pathlib import Path
import warnings

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import pytest

optiwindnet = pytest.importorskip("optiwindnet")

import ard.utils.io
import ard.utils.test_utils
import ard.collection.optiwindnet_wrap as ard_own


def make_modeling_options(x_turbines, y_turbines, x_substations, y_substations):

    # set up the modeling options
    N_turbines = len(x_turbines)
    N_substations = len(x_substations)
    modeling_options = {
        "windIO_plant": {
            "wind_farm": {
                "electrical_substations": [
                    {
                        "electrical_substation": {
                            "coordinates": {"x": xv, "y": yv},
                        },
                    }
                    for xv, yv in zip(x_substations, y_substations)
                ],
            },
        },
        "layout": {
            "N_turbines": N_turbines,
            "N_substations": N_substations,
            "x_turbines": x_turbines,
            "y_turbines": y_turbines,
        },
        "collection": {
            "max_turbines_per_string": 8,
            "model_options": dict(
                topology="branched",
                feeder_route="segmented",
                feeder_limit="unlimited",
            ),
            "solver_name": "highs",
            "solver_options": dict(
                time_limit=10,
                mip_gap=0.005,  # TODO ???
            ),
        },
    }

    return modeling_options


@pytest.mark.usefixtures("subtests")
class TestOptiWindNetCollection:

    def setup_method(self):

        # create the farm layout specification
        n_turbines = 25
        x_turbines, y_turbines = [
            130.0 * 7 * v.flatten()
            for v in np.meshgrid(
                np.linspace(-2, 2, int(np.sqrt(n_turbines)), dtype=int),
                np.linspace(-2, 2, int(np.sqrt(n_turbines)), dtype=int),
            )
        ]
        x_substations = np.array([-500.0, 500.0], dtype=np.float64)
        y_substations = np.array([-500.0, 500.0], dtype=np.float64)

        modeling_options = make_modeling_options(
            x_turbines=x_turbines,
            y_turbines=y_turbines,
            x_substations=x_substations,
            y_substations=y_substations,
        )

        # create the OpenMDAO model
        model = om.Group()
        self.collection = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_modeling(self, subtests):
        """
        make sure the modeling_options has what we need for farmaero
        """

        with subtests.test("modeling_options"):
            assert "modeling_options" in [k for k, _ in self.collection.options.items()]
        with subtests.test("layout"):
            assert "layout" in self.collection.options["modeling_options"].keys()
        with subtests.test("N_turbines"):
            assert (
                "N_turbines"
                in self.collection.options["modeling_options"]["layout"].keys()
            )
        with subtests.test("N_substations"):
            assert (
                "N_substations"
                in self.collection.options["modeling_options"]["layout"].keys()
            )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the inputs in the component match what we planned
            input_list = [k for k, v in self.collection.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ]:
                with subtests.test("inputs"):
                    assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.collection.list_outputs()]
            for var_to_check in [
                "terse_links",
                "total_length_cables",
            ]:
                assert var_to_check in output_list

            # make sure that the outputs in the component match what we planned
            discrete_output_list = [
                k for k, v in self.collection._discrete_outputs.items()
            ]
            for var_to_check in [
                "length_cables",
                "load_cables",
                "max_load_cables",
            ]:
                assert var_to_check in discrete_output_list

    def test_compute_pyrite(self, subtests):

        # run optiwindnet
        self.prob.run_model()

        # collect data to validate
        validation_data = {
            "terse_links": self.prob.get_val("collection.terse_links"),
            "length_cables": self.prob.get_val("collection.length_cables"),
            "load_cables": self.prob.get_val("collection.load_cables"),
            "total_length_cables": self.prob.get_val("collection.total_length_cables"),
            "max_load_cables": self.prob.get_val("collection.max_load_cables"),
        }

        # validate data against pyrite file
        pyrite_data = ard.utils.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_optiwindnet_pyrite.npz",
            # rtol_val=5e-3, # only for check in validator
            #  rewrite=True,  # uncomment to write new pyrite file
            load_only=True,
        )

        for key in validation_data:
            with subtests.test(key):
                assert np.allclose(validation_data[key], pyrite_data[key], rtol=5e-3)


class TestOptiWindNetCollection12Turbines:

    def setup_method(self):

        x_turbines = np.array(
            [1940, 1920, 1475, 1839, 1277, 442, 737, 1060, 522, 87, 184, 71],
            dtype=np.float64,
        )
        y_turbines = np.array(
            [279, 703, 696, 1250, 1296, 1359, 435, 26, 176, 35, 417, 878],
            dtype=np.float64,
        )
        x_substations = np.array([696], dtype=np.float64)
        y_substations = np.array([1063], dtype=np.float64)

        self.modeling_options = make_modeling_options(
            x_turbines=x_turbines,
            y_turbines=y_turbines,
            x_substations=x_substations,
            y_substations=y_substations,
        )

        # create the OpenMDAO model
        model = om.Group()
        self.collection = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_example_location(self):

        # deep copy modeling options and adjust
        modeling_options = self.modeling_options
        modeling_options["collection"]["max_turbines_per_string"] = 4

        # create the OpenMDAO model
        model = om.Group()
        collection_example = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )
        prob = om.Problem(model)
        prob.setup()

        prob.set_val(
            "collection.x_border",
            np.array(
                [1951, 1951, 386, 650, 624, 4, 4, 1152, 917, 957], dtype=np.float64
            ),
        )
        prob.set_val(
            "collection.y_border",
            np.array(
                [200, 1383, 1383, 708, 678, 1036, 3, 3, 819, 854], dtype=np.float64
            ),
        )

        # run optiwindnet
        prob.run_model()

        assert (
            abs(prob.get_val("collection.total_length_cables") - 6564.7653295074515)
            < 1e-7
        )
        cpJ = prob.check_partials(out_stream=None)
        assert_check_partials(cpJ, atol=1.0e-5, rtol=1.0e-3)


class TestOptiWindNetCollection5Turbines:

    def setup_method(self):
        n_turbines = 5
        theta_turbines = np.linspace(0.0, 2 * np.pi, n_turbines + 1)[:-1]
        x_turbines = 7.0 * 130.0 * np.sin(theta_turbines)
        y_turbines = 7.0 * 130.0 * np.cos(theta_turbines)
        x_substations = np.array([0.0])
        y_substations = np.array([0.0])
        self.modeling_options = make_modeling_options(
            x_turbines, y_turbines, x_substations, y_substations
        )

        # create the OpenMDAO model
        model = om.Group()
        self.collection = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_compute_partials_mini_pentagon(self):
        """
        run a really small case so that qualititative changes do not occur s.t.
        we can validate the differences using the OM built-ins; use a pentagon
        with a centered substation so there is no chaining.
        """

        # deep copy modeling options and adjust
        modeling_options = copy.deepcopy(self.modeling_options)
        modeling_options["layout"]["N_turbines"] = 5
        modeling_options["layout"]["N_substations"] = 1
        modeling_options["windIO_plant"]["wind_farm"]["electrical_substations"] = [
            {
                "electrical_substation": {
                    "coordinates": {"x": xv, "y": yv},
                },
            }
            for xv, yv in zip([0.0], [0.0])
        ]

        # create the OpenMDAO model
        model = om.Group()
        collection_mini = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        prob = om.Problem(model)
        prob.setup()

        # run optiwindnet
        prob.run_model()

        if False:  # for hand-debugging
            J0 = prob.compute_totals(
                "collection.length_cables", "collection.x_turbines"
            )
            prob.model.approx_totals()
            J0p = prob.compute_totals(
                "collection.length_cables", "collection.x_turbines"
            )

            print("J0:")
            print(J0)
            print("\n\n\n\n\nJ0p:")
            print(J0p)

            assert False

        # automated OpenMDAO fails because it re-runs the network work
        cpJ = prob.check_partials(out_stream=None)
        assert_check_partials(cpJ, atol=1.0e-5, rtol=1.0e-3)

    def test_compute_partials_mini_line(self):
        """
        run a really small case so that qualititative changes do not occur s.t.
        we can validate the differences using the OM built-ins; use a linear
        layout with a continuing substation so there is no variation.
        """

        # deep copy modeling options and adjust
        modeling_options = copy.deepcopy(self.modeling_options)
        modeling_options["layout"]["N_turbines"] = 5
        modeling_options["layout"]["N_substations"] = 1
        modeling_options["windIO_plant"]["wind_farm"]["electrical_substations"] = [
            {
                "electrical_substation": {
                    "coordinates": {"x": xv, "y": yv},
                },
            }
            for xv, yv in zip([5.0], [5.0])
        ]

        # create the OpenMDAO model
        model = om.Group()
        collection_mini = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        prob = om.Problem(model)
        prob.setup()
        # set in the variables
        s_turbines = np.array([1, 2, 3, 4, 5])
        X_turbines = 7.0 * 130.0 * s_turbines
        Y_turbines = np.log(7.0 * 130.0 * s_turbines)
        X_substations = np.array([-3.5 * 130.0])
        Y_substations = np.array([-3.5 * 130.0])
        prob.set_val("collection.x_turbines", X_turbines)
        prob.set_val("collection.y_turbines", Y_turbines)
        prob.set_val("collection.x_substations", X_substations)
        prob.set_val("collection.y_substations", Y_substations)

        # run optiwindnet
        prob.run_model()

        if False:  # for hand-debugging
            J0 = prob.compute_totals(
                "collection.length_cables", "collection.x_turbines"
            )
            prob.model.approx_totals()
            J0p = prob.compute_totals(
                "collection.length_cables", "collection.x_turbines"
            )

            print("J0:")
            print(J0)
            print("\n\n\n\n\nJ0p:")
            print(J0p)

            assert False

        # automated OpenMDAO fails because it re-runs the network work
        cpJ = prob.check_partials(out_stream=None)
        assert_check_partials(cpJ, atol=1.0e-5, rtol=1.0e-3)


class TestOptiWindNetCollection4TurbinesOverlap:

    def setup_method(self):
        self.n_turbines = 4
        self.x_turbines = 7.0 * 130.0 * np.array([-1.0, 0.0, 0.0, 1.0])
        self.y_turbines = 7.0 * 130.0 * np.array([-1.0, 0.0, 0.0, 1.0])
        self.x_substations = np.array([-100.0])
        self.y_substations = np.array([100.0])
        self.modeling_options = make_modeling_options(
            self.x_turbines,
            self.y_turbines,
            self.x_substations,
            self.y_substations,
        )

        # create the OpenMDAO model
        model = om.Group()
        self.collection = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_perturbation_and_warning(self, subtests):
        """
        Test that OptiwindnetCollection issues a warning when turbines and/or substations
        have coincident coordinates, but still produces valid results.

        This test verifies that:
        1. A warning is raised matching the pattern about coincident turbines/substations
        2. The model still executes successfully despite the warning
        3. The calculated total cable length matches the expected reference value
        """

        # deep copy modeling options and adjust
        modeling_options = self.modeling_options

        # create the OpenMDAO model
        model = om.Group()
        collection_mini = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        prob = om.Problem(model)
        prob.setup()

        prob.set_val("collection.x_turbines", self.x_turbines)
        prob.set_val("collection.y_turbines", self.y_turbines)
        prob.set_val("collection.x_substations", self.x_substations)
        prob.set_val("collection.y_substations", self.y_substations)

        with subtests.test("warn on duplicate turbine"):
            with pytest.warns(
                match=r"coincident turbines and/or substations in optiwindnet setup"
            ) as warning:
                # run optiwindnet
                prob.run_model()
                for w in warning:
                    print(w.message)

        # make sure that it still runs and we match a reference value
        total_length_cables_reference = 2715.29003976
        with subtests.test("match reference value"):
            assert np.isclose(
                prob.get_val("collection.total_length_cables"),
                total_length_cables_reference,
            )

        # make sure the values in the optiwindnet graph are each close to a
        # turbine but also include perturbation where it should be
        T = collection_mini.graph.graph["T"]
        R = collection_mini.graph.graph["R"]
        VertexC = np.array(collection_mini.graph.graph["VertexC"])

        for idx_T, xy_VertexCT in enumerate(VertexC[:T]):
            # check if this turbine coordinate matches a turbine position (exactly or with perturbation)
            matches_exactly = np.any(
                np.logical_and(
                    xy_VertexCT[0] == self.x_turbines,
                    xy_VertexCT[1] == self.y_turbines,
                )
            )
            matches_with_perturbation = np.any(
                np.logical_and(
                    np.isclose(xy_VertexCT[0], self.x_turbines, atol=1e-2)
                    & (xy_VertexCT[0] != self.x_turbines),
                    np.isclose(xy_VertexCT[1], self.y_turbines, atol=1e-2)
                    & (xy_VertexCT[1] != self.y_turbines),
                )
            )
            with subtests.test(f"turbine {idx_T} exact xor perturbed"):
                assert matches_exactly ^ matches_with_perturbation  # boolean xor

        for xy_VertexCR in VertexC[-R:]:
            # check if this turbine coordinate matches a turbine position (exactly or with perturbation)
            matches_exactly = np.any(
                np.logical_and(
                    xy_VertexCR[0] == self.x_substations,
                    xy_VertexCR[1] == self.y_substations,
                )
            )
            matches_with_perturbation = np.any(
                np.logical_and(
                    np.isclose(xy_VertexCR[0], self.x_substations, atol=1e-2)
                    & (xy_VertexCR[0] != self.x_substations),
                    np.isclose(xy_VertexCR[1], self.y_substations, atol=1e-2)
                    & (xy_VertexCR[1] != self.y_substations),
                )
            )
            with subtests.test(f"substation {idx_T} exact xor perturbed"):
                assert matches_exactly ^ matches_with_perturbation  # boolean xor

    def test_nowarning(self, subtests):
        """
        Test that no warnings are raised when turbines are positioned without overlap.

        This test verifies that the OptiwindnetCollection component runs without
        warnings when turbine positions are adjusted to avoid overlap. It repositions
        turbines along a linear interpolation between the first and last turbine
        positions, then runs the model with warnings set to raise errors. The test
        also validates that the total cable length calculation produces the expected
        reference value.
        """

        # deep copy modeling options and adjust
        modeling_options = self.modeling_options

        # create the OpenMDAO model
        model = om.Group()
        collection_mini = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        # move the turbines so they don't overlap anymore
        self.x_turbines[1] = 0.3 * self.x_turbines[0] + 0.7 * self.x_turbines[-1]
        self.y_turbines[1] = 0.3 * self.y_turbines[0] + 0.7 * self.y_turbines[-1]
        self.x_turbines[2] = 0.7 * self.x_turbines[0] + 0.3 * self.x_turbines[-1]
        self.y_turbines[2] = 0.7 * self.y_turbines[0] + 0.3 * self.y_turbines[-1]

        prob = om.Problem(model)
        prob.setup()

        prob.set_val("collection.x_turbines", self.x_turbines)
        prob.set_val("collection.y_turbines", self.y_turbines)
        prob.set_val("collection.x_substations", self.x_substations)
        prob.set_val("collection.y_substations", self.y_substations)

        # make sure no warnings occur when the turbines don't overlap
        with subtests.test("no warning for unique turbines/substations"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # run optiwindnet
                prob.run_model()

        # make sure that it still runs and we match a reference value
        total_length_cables_reference = 2612.01404984
        with subtests.test("match reference value"):
            assert np.isclose(
                prob.get_val("collection.total_length_cables"),
                total_length_cables_reference,
            )


class TestOptiWindNetCollectionSubstationOverlap:

    def setup_method(self):
        self.n_turbines = 4
        self.x_turbines = 7.0 * 130.0 * np.array([-1.0, 0.0, 1.0, 2.0])
        self.y_turbines = 7.0 * 130.0 * np.array([-1.0, 0.0, 1.0, 2.0])
        self.x_substations = np.array([0.0])
        self.y_substations = np.array([0.0])
        self.modeling_options = make_modeling_options(
            self.x_turbines,
            self.y_turbines,
            self.x_substations,
            self.y_substations,
        )

        # create the OpenMDAO model
        model = om.Group()
        self.collection = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_perturbation_and_warning(self, subtests):
        """
        Test that OptiwindnetCollection issues a warning when turbines and/or substations
        have coincident coordinates, but still produces valid results.

        This test verifies that:
        1. A warning is raised matching the pattern about coincident turbines/substations
        2. The model still executes successfully despite the warning
        3. The calculated total cable length matches the expected reference value
        """

        # deep copy modeling options and adjust
        modeling_options = self.modeling_options

        # create the OpenMDAO model
        model = om.Group()
        collection_mini = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        prob = om.Problem(model)
        prob.setup()

        prob.set_val("collection.x_turbines", self.x_turbines)
        prob.set_val("collection.y_turbines", self.y_turbines)
        prob.set_val("collection.x_substations", self.x_substations)
        prob.set_val("collection.y_substations", self.y_substations)

        with subtests.test("warn on turbine/substation intersection"):
            with pytest.warns(
                match=r"coincident turbines and/or substations in optiwindnet setup"
            ) as warning:
                # run optiwindnet
                prob.run_model()
                for w in warning:
                    print(w.message)

        # make sure that it still runs and we match a reference value
        total_length_cables_reference = 3860.80302628
        with subtests.test("match reference value"):
            assert np.isclose(
                prob.get_val("collection.total_length_cables"),
                total_length_cables_reference,
            )

        # make sure the values in the optiwindnet graph are each close to a
        # turbine but also include perturbation where it should be
        T = collection_mini.graph.graph["T"]
        R = collection_mini.graph.graph["R"]
        VertexC = np.array(collection_mini.graph.graph["VertexC"])

        for idx_T, xy_VertexCT in enumerate(VertexC[:T]):
            # check if this turbine coordinate matches a turbine position (exactly or with perturbation)
            matches_exactly = np.any(
                np.logical_and(
                    xy_VertexCT[0] == self.x_turbines,
                    xy_VertexCT[1] == self.y_turbines,
                )
            )
            matches_with_perturbation = np.any(
                np.logical_and(
                    np.isclose(xy_VertexCT[0], self.x_turbines, atol=1e-2)
                    & (xy_VertexCT[0] != self.x_turbines),
                    np.isclose(xy_VertexCT[1], self.y_turbines, atol=1e-2)
                    & (xy_VertexCT[1] != self.y_turbines),
                )
            )
            with subtests.test(f"turbine {idx_T} exact xor perturbed"):
                assert matches_exactly ^ matches_with_perturbation  # xor

        for idx_R, xy_VertexCR in enumerate(VertexC[-R:]):
            # check if this turbine coordinate matches a turbine position (exactly or with perturbation)
            matches_exactly = np.any(
                np.logical_and(
                    xy_VertexCR[0] == self.x_substations,
                    xy_VertexCR[1] == self.y_substations,
                )
            )
            matches_with_perturbation = np.any(
                np.logical_and(
                    np.isclose(xy_VertexCR[0], self.x_substations, atol=1e-2)
                    & (xy_VertexCR[0] != self.x_substations),
                    np.isclose(xy_VertexCR[1], self.y_substations, atol=1e-2)
                    & (xy_VertexCR[1] != self.y_substations),
                )
            )
            with subtests.test(f"substation {idx_T} exact xor perturbed"):
                assert matches_exactly ^ matches_with_perturbation  # xor

    def test_nowarning(self, subtests):
        """
        Test that no warnings are raised when turbines are positioned without overlap.

        This test verifies that the OptiwindnetCollection component runs without
        warnings when turbine positions are adjusted to avoid overlap. It repositions
        turbines along a linear interpolation between the first and last turbine
        positions, then runs the model with warnings set to raise errors. The test
        also validates that the total cable length calculation produces the expected
        reference value.
        """

        # deep copy modeling options and adjust
        modeling_options = self.modeling_options

        # create the OpenMDAO model
        model = om.Group()
        collection_mini = model.add_subsystem(
            "collection",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        # move the substations so they don't overlap anymore
        self.x_substations[0] = 100.0
        self.y_substations[0] = 100.0

        prob = om.Problem(model)
        prob.setup()

        prob.set_val("collection.x_turbines", self.x_turbines)
        prob.set_val("collection.y_turbines", self.y_turbines)
        prob.set_val("collection.x_substations", self.x_substations)
        prob.set_val("collection.y_substations", self.y_substations)

        # make sure no warnings occur when the turbines don't overlap
        with subtests.test("no warning for unique turbines/substations"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # run optiwindnet
                prob.run_model()

        # make sure that it still runs and we match a reference value
        total_length_cables_reference = 3860.80302528
        with subtests.test("match reference value"):
            assert np.isclose(
                prob.get_val("collection.total_length_cables"),
                total_length_cables_reference,
            )
