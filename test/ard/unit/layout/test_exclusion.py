import numpy as np
import openmdao.api as om

import pytest

import ard.layout.exclusions as exclusions


@pytest.mark.usefixtures("subtests")
class TestFarmExclusionDistancePolygon:
    """
    Test the FarmExclusionDistancePolygon component.
    """

    def setup_method(self):

        self.D_rotor = 100.0

        # set turbine layout (3x3 grid 5D spacing)
        X, Y = [
            4.0 * self.D_rotor * v
            for v in np.meshgrid(np.arange(0, 3), np.arange(0, 3))
        ]

        self.x_turbines = X.flatten()
        self.y_turbines = Y.flatten()

        self.N_turbines = len(self.x_turbines)

    def test_single_trapezoid_distance(self):

        # set modeling options
        ex = [0.0, 600.0, 1000.0, 1000.0]
        ey = [0.0, 0.0, 400.0, 1000.0]
        modeling_options_single = {
            "windIO_plant": {
                "name": "unit test dummy",
                "site": {
                    "name": "unit test site",
                    "exclusions": {
                        "polygons": [
                            {
                                "x": ex,
                                "y": ey,
                            },
                        ],
                    },
                },
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    }
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
            },
        }

        # set up openmdao problem
        model_single = om.Group()
        model_single.add_subsystem(
            "exclusions",
            exclusions.FarmExclusionDistancePolygon(
                modeling_options=modeling_options_single,
            ),
            promotes=["*"],
        )
        prob_single = om.Problem(model_single)
        prob_single.setup()

        prob_single.set_val("x_turbines", self.x_turbines)
        prob_single.set_val("y_turbines", self.y_turbines)

        prob_single.run_model()

        expected_distances = np.array(
            [
                0.000000000000,
                0.000000000000,
                -141.4213562373095,
                -282.842712474619,
                0.000000000000,
                141.4213562373095,
                -565.685424949238,
                -282.842712474619,
                0.0000000000000,
            ]
        )
        assert np.allclose(
            prob_single["exclusion_distances"], expected_distances, atol=1e-3
        )

    def test_single_trapezoid_derivatives(self, subtests):

        # set modeling options
        modeling_options_single = {
            "windIO_plant": {
                "name": "unit test dummy",
                "site": {
                    "name": "unit test site",
                    "exclusions": {
                        "polygons": [
                            {
                                "x": [0.0, 600.0, 1000.0, 1000.0],
                                "y": [0.0, 0.0, 400.0, 1000.0],
                            },
                        ],
                    },
                },
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    },
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
            },
        }

        # set up openmdao problem
        model_single = om.Group()
        model_single.add_subsystem(
            "exclusions",
            exclusions.FarmExclusionDistancePolygon(
                modeling_options=modeling_options_single,
            ),
            promotes=["*"],
        )
        prob_single = om.Problem(model_single)
        prob_single.setup()

        prob_single.set_val("x_turbines", self.x_turbines)
        prob_single.set_val("y_turbines", self.y_turbines)

        prob_single.run_model()

        derivatives_computed = prob_single.compute_totals(
            of=["exclusion_distances"],
            wrt=["x_turbines", "y_turbines"],
        )

        derivatives_expected = {
            ("exclusion_distances", "x_turbines"): np.array(
                [
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        -0.70710678,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.70710678,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        -0.70710678,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.70710678,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.70710678,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                ]
            ),
            ("exclusion_distances", "y_turbines"): np.array(
                [
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.70710678,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        -0.70710678,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.70710678,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        -0.70710678,
                        0.00000000,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        -0.70710678,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                        0.00000000,
                    ],
                ]
            ),
        }

        # assert a match
        with subtests.test("wrt x_turbines"):
            assert np.allclose(
                derivatives_computed[("exclusion_distances", "x_turbines")],
                derivatives_expected[("exclusion_distances", "x_turbines")],
                atol=1e-3,
            )
        with subtests.test("wrt y_turbines"):
            assert np.allclose(
                derivatives_computed[("exclusion_distances", "y_turbines")],
                derivatives_expected[("exclusion_distances", "y_turbines")],
                atol=1e-3,
            )

    def test_single_triangle_distance(self):

        # set modeling options
        modeling_options_single = {
            "windIO_plant": {
                "name": "unit test dummy",
                "site": {
                    "name": "unit test site",
                    "exclusions": {
                        "polygons": [
                            {
                                "x": [0.0, 800.0, 0.0],
                                "y": [200.0, 1000.0, 1000.0],
                            },
                        ],
                    },
                },
                "wind_farm": {
                    "turbine": {
                        "rotor_diameter": self.D_rotor,
                    }
                },
            },
            "layout": {
                "N_turbines": self.N_turbines,
            },
        }

        # set up openmdao problem
        model_single = om.Group()
        model_single.add_subsystem(
            "exclusions",
            exclusions.FarmExclusionDistancePolygon(
                modeling_options=modeling_options_single,
            ),
            promotes=["*"],
        )
        prob_single = om.Problem(model_single)
        prob_single.setup()

        prob_single.set_val("x_turbines", self.x_turbines)
        prob_single.set_val("y_turbines", self.y_turbines)

        prob_single.run_model()

        expected_distances = np.array(
            [
                -200.000000000000,  # 0
                -424.2640687119286,  # 1
                -707.1067811865476,  # 2
                0.000000000000,  # 3
                -141.4213562373095,  # 4
                -424.2640687119286,  # 5
                0.000000000000,  # 6
                141.4213562373095,  # 7
                -141.4213562373095,  # 8
            ]
        )

        assert np.allclose(
            prob_single["exclusion_distances"], expected_distances, atol=1e-3
        )


#     def test_multi_polygon_distance(self):
#
#         boundary_vertices_0 = np.array(
#             [
#                 [0.0, 0.0],
#                 [600.0, 0.0],
#                 [1000.0, 400.0],
#                 [1000.0, 1000.0],
#             ]
#         )
#
#         boundary_vertices_1 = np.array(
#             [
#                 [0.0, 200.0],
#                 [800.0, 1000.0],
#                 [0.0, 1000.0],
#             ]
#         )
#
#         region_assignments = np.ones(self.N_turbines, dtype=int)
#         region_assignments[0:3] = 0
#
#         # set modeling options
#         modeling_options_multi = {
#             "windIO_plant": {
#                 "name": "unit test dummy",
#                 "site": {
#                     "name": "unit test site",
#                     "exclusions": {
#                         "polygons": [
#                             {
#                                 "x": boundary_vertices_0[:, 0].tolist(),
#                                 "y": boundary_vertices_0[:, 1].tolist(),
#                             },
#                             {
#                                 "x": boundary_vertices_1[:, 0].tolist(),
#                                 "y": boundary_vertices_1[:, 1].tolist(),
#                             },
#                         ]
#                     },
#                 },
#                 "wind_farm": {
#                     "turbine": {
#                         "rotor_diameter": self.D_rotor,
#                     }
#                 },
#             },
#             "layout": {
#                 "N_turbines": self.N_turbines,
#             },
#             "exclusions": {
#                 "turbine_exclusion_assignments": region_assignments,
#             },
#         }
#
#         # set up openmdao problem
#         model = om.Group()
#         model.add_subsystem(
#             "exclusions",
#             exclusions.FarmExclusionDistancePolygon(
#                 modeling_options=modeling_options_multi,
#             ),
#             promotes=["*"],
#         )
#         prob = om.Problem(model)
#         prob.setup()
#
#         prob.set_val("x_turbines", self.x_turbines)
#         prob.set_val("y_turbines", self.y_turbines)
#
#         prob.run_model()
#
#         print(f"DEBUG!!!!! x_turbines: {self.x_turbines}")
#         print(f"DEBUG!!!!! y_turbines: {self.y_turbines}")
#         print(f"DEBUG!!!!! region_assignments: {region_assignments}")
#
#         expected_distances = np.array(
#             [
#                 0.000000000000,
#                 0.000000000000,
#                 -141.4213562373095,
#                 0.000000000000,
#                 -141.4213562373095,
#                 -424.2640687119286,
#                 0.000000000000,
#                 141.4213562373095,
#                 -141.4213562373095,
#             ]
#         )
#
#         # assert a match: loose tolerance for turbines in corners due to using the smooth min
#         assert np.allclose(prob["exclusion_distances"], expected_distances, atol=1e-2)
#
#     def test_multi_polygon_derivatives(self, subtests):
#
#         boundary_vertices_0 = np.array(
#             [
#                 [0.0, 0.0],
#                 [1000.0, 0.0],
#                 [1000.0, 200.0],
#                 [0.0, 200.0],
#             ]
#         )
#
#         boundary_vertices_1 = np.array(
#             [
#                 [0.0, 300.0],
#                 [1000.0, 300.0],
#                 [1000.0, 1000.0],
#                 [1100.0, 1100.0],
#                 [0.0, 1100.0],
#             ]
#         )
#
#         region_assignments = np.ones(self.N_turbines, dtype=int)
#         region_assignments[0:3] = 0
#
#         # set modeling options
#         modeling_options_multi = {
#             "windIO_plant": {
#                 "name": "unit test dummy",
#                 "site": {
#                     "name": "unit test site",
#                     "exclusions": {
#                         "polygons": [
#                             {
#                                 "x": boundary_vertices_0[:, 0].tolist(),
#                                 "y": boundary_vertices_0[:, 1].tolist(),
#                             },
#                             {
#                                 "x": boundary_vertices_1[:, 0].tolist(),
#                                 "y": boundary_vertices_1[:, 1].tolist(),
#                             },
#                         ]
#                     },
#                 },
#                 "wind_farm": {
#                     "turbine": {
#                         "rotor_diameter": self.D_rotor,
#                     }
#                 },
#             },
#             "layout": {
#                 "N_turbines": self.N_turbines,
#             },
#             "exclusions": {
#                 "turbine_exclusion_assignments": region_assignments,
#             },
#         }
#
#         # set up openmdao problem
#         model = om.Group()
#         model.add_subsystem(
#             "exclusions",
#             exclusions.FarmExclusionDistancePolygon(
#                 modeling_options=modeling_options_multi,
#             ),
#             promotes=["*"],
#         )
#         prob = om.Problem(model)
#         prob.setup()
#
#         prob.set_val("x_turbines", self.x_turbines)
#         prob.set_val("y_turbines", self.y_turbines)
#
#         prob.run_model()
#
#         derivatives_computed = prob.compute_totals(
#             of=["exclusion_distances"],
#             wrt=["x_turbines", "y_turbines"],
#         )
#
#         derivatives_expected = {
#             ("exclusion_distances", "x_turbines"): -np.array(
#                 [
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#                 ]
#             ),
#             ("exclusion_distances", "y_turbines"): -np.array(
#                 [
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 ]
#             ),
#         }
#
#         # assert a match
#         with subtests.test("wrt x_turbines"):
#             assert np.allclose(
#                 derivatives_computed[("exclusion_distances", "x_turbines")],
#                 derivatives_expected[("exclusion_distances", "x_turbines")],
#                 atol=1e-3,
#             )
#         with subtests.test("wrt y_turbines"):
#             assert np.allclose(
#                 derivatives_computed[("exclusion_distances", "y_turbines")],
#                 derivatives_expected[("exclusion_distances", "y_turbines")],
#                 atol=1e-3,
#             )
