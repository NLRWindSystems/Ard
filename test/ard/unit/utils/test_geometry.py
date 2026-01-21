import numpy as np
import jax.test_util
import ard.utils.geometry as geo_utils
import pytest


@pytest.mark.usefixtures("subtests")
class TestPadPolygon:

    def test_zero_vertex(self, subtests):
        boundary_vertices = [
            np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 200.0], [0.0, 200.0]]),
            np.array(
                [
                    [0.0, 300.0],
                    [1000.0, 300.0],
                    [1000.0, 1000.0],
                    [1100.0, 1100.0],
                    [0.0, 1100.0],
                ]
            ),
        ]

        padded_expected_vertices = [
            np.array(
                [[0.0, 0.0], [1000.0, 0.0], [1000.0, 200.0], [0.0, 200.0], [0.0, 200.0]]
            ),
            np.array(
                [
                    [0.0, 300.0],
                    [1000.0, 300.0],
                    [1000.0, 1000.0],
                    [1100.0, 1100.0],
                    [0.0, 1100.0],
                ]
            ),
        ]

        max_vertices = max(len(polygon) for polygon in boundary_vertices)

        with subtests.test("boundary 0"):
            paded_polygon = geo_utils.pad_polygon(boundary_vertices[0], max_vertices)
            assert np.allclose(paded_polygon, padded_expected_vertices[0])

        with subtests.test("boundary 1"):
            paded_polygon = geo_utils.pad_polygon(boundary_vertices[1], max_vertices)
            assert np.allclose(paded_polygon, padded_expected_vertices[1])


@pytest.mark.usefixtures("subtests")
class TestGetNearestPolygons:
    """
    Test for get_nearest_polygons function
    """

    def test_get_nearest_polygons_single_region(self):

        points = np.array([[0.25, 0.5], [1.5, 0.5]])
        polygons = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        expected_regions = np.array([0, 0])

        test_result = geo_utils.get_nearest_polygons(
            boundary_vertices=[polygons],
            points_x=points[:, 0],
            points_y=points[:, 1],
        )

        assert np.allclose(test_result, expected_regions)

    def test_get_nearest_polygons_multi_region(self):

        points = np.array([[0.25, 0.5], [1.95, 0.5]], dtype=float)
        polygons = [
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([[1, 0], [2, 0], [2, 1]], dtype=float),
        ]
        expected_regions = np.array([0, 1], dtype=int)

        test_result = geo_utils.get_nearest_polygons(
            boundary_vertices=polygons,
            points_x=points[:, 0],
            points_y=points[:, 1],
        )

        assert np.allclose(test_result, expected_regions)


@pytest.mark.usefixtures("subtests")
class TestDistancePointToMultiPolygonRayCasting:
    """
    Test for distance_point_to_polygon_ray_casting
    """

    def setup_method(self):
        self.distance_multi_point_to_multi_polygon_ray_casting_jac = jax.jacrev(
            geo_utils.distance_multi_point_to_multi_polygon_ray_casting, [0, 1]
        )
        pass

    def test_distance_multi_point_to_multi_polygon_inside_outside_single_square_region(
        self,
    ):

        points = np.array([[0.25, 0.5], [1.5, 0.5]])
        polygons = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        expected_distance = [-0.25, 0.5]

        test_result = geo_utils.distance_multi_point_to_multi_polygon_ray_casting(
            boundary_vertices=[polygons],
            points_x=points[:, 0],
            points_y=points[:, 1],
            regions=np.array([0, 0]),
        )

        assert np.allclose(test_result, expected_distance)

    def test_distance_multi_point_to_multi_polygon_inside_outside_single_offsettriangle_region(
        self,
    ):

        points = np.array(
            [
                [0.2, 0.6],
                [0.6, 0.2],
                [0.4, 0.8],
                [0.8, 0.8],
            ]
        )
        polygons = [
            np.array(
                [
                    [0.0, 0.2],
                    [0.8, 1.0],
                    [0.0, 1.0],
                ]
            )
        ]

        expected_distance = [
            -0.1 * np.sqrt(2),
            0.3 * np.sqrt(2),
            -0.1 * np.sqrt(2),
            0.1 * np.sqrt(2),
        ]

        test_result = geo_utils.distance_multi_point_to_multi_polygon_ray_casting(
            boundary_vertices=polygons,
            points_x=points[:, 0],
            points_y=points[:, 1],
            regions=np.array([0, 0, 0, 0]),
        )

        assert np.allclose(test_result, expected_distance)

    def test_distance_multi_point_to_multi_polygon_inside_outside_multiple_regions(
        self,
    ):

        points = np.array([[0.25, 0.5], [2.5, 0.5]])
        polygons = [
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            np.array([[1, 0], [2, 0], [2, 1]]),
        ]

        expected_distance = [-0.25, 0.5]

        test_result = geo_utils.distance_multi_point_to_multi_polygon_ray_casting(
            boundary_vertices=polygons,
            points_x=points[:, 0],
            points_y=points[:, 1],
            regions=np.array([0, 1]),
        )

        assert np.allclose(test_result, expected_distance)

    def test_distance_multi_point_to_multi_polygon_inside_outside_multiple_regions_jac(
        self, subtests
    ):

        points = np.array([[0.25, 0.5], [1.95, 0.5]], dtype=float)
        polygons = [
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([[1, 0], [2, 0], [2, 1]], dtype=float),
        ]
        regions = np.array([0, 1], dtype=int)

        test_result = self.distance_multi_point_to_multi_polygon_ray_casting_jac(
            points[:, 0], points[:, 1], boundary_vertices=polygons, regions=regions
        )

        dd1d2_dx = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float)
        dd1d2_dy = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)

        expected_result = np.array([dd1d2_dx, dd1d2_dy], dtype=float)

        with subtests.test("analytic derivatives"):
            # note that the distance should get more negative as the point moves deeper inside the polygon
            assert np.allclose(test_result, expected_result)

        with subtests.test("numeric derivatives"):

            def grad_check_func(points_x, point_y):
                return geo_utils.distance_multi_point_to_multi_polygon_ray_casting(
                    points_x, point_y, polygons, regions
                )

            try:
                jax.test_util.check_grads(
                    grad_check_func,
                    args=(points[:, 0], points[:, 1]),
                    order=1,
                    rtol=1e-3,
                )
            except AssertionError:
                pytest.fail(
                    "Unexpected AssertionError when checking gradients, gradients may be incorrect"
                )


@pytest.mark.usefixtures("subtests")
class TestDistancePointToPolygonRayCasting:
    """
    Test for distance_point_to_polygon_ray_casting
    """

    def setup_method(self):
        self.distance_point_to_polygon_ray_casting_grad = jax.grad(
            geo_utils.distance_point_to_polygon_ray_casting, [0]
        )
        pass

    def test_distance_point_to_unitsquare_inside(self, subtests):

        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        for idx_point, (point, expected_distance) in enumerate(
            [(np.array([0.25, 0.5]), -0.25)]
        ):

            with subtests.test(f"point {idx_point}"):
                test_result = geo_utils.distance_point_to_polygon_ray_casting(
                    point, vertices=polygon
                )
                assert test_result == pytest.approx(expected_distance)

    def test_distance_point_to_offsettriangle_inside(self, subtests):

        polygon = np.array([[0.0, 0.2], [0.8, 1.0], [0.0, 1.0]])

        for idx_point, (point, expected_distance) in enumerate(
            [
                (np.array([0.2, 0.6]), -0.14142135623730953),
                (np.array([0.4, 0.8]), -0.14142135623730953),
                (np.array([0.8, 0.8]), 0.14142135623730953),
            ]
        ):

            with subtests.test(f"point {idx_point}"):
                test_result = geo_utils.distance_point_to_polygon_ray_casting(
                    point, vertices=polygon
                )
                assert test_result == pytest.approx(expected_distance)

    def test_process_edge_multiple(self, subtests):

        polygon = np.array([[0.0, 0.2], [0.8, 1.0], [0.0, 1.0]])
        point = np.array([0.8, 0.8])
        expected_below = [False, False, True]

        for idx_end_point, end_point in enumerate(polygon):

            with subtests.test(f"end point {idx_end_point}"):
                start_point = polygon[idx_end_point - 1]
                is_below, distance, vertex_crossing = geo_utils.process_edge(
                    edge_start=start_point, edge_end=end_point, point=point, shift=1e-10
                )
                assert is_below == expected_below[idx_end_point]

    def test_process_edge_multiple_above(self, subtests):

        polygon = np.array([[0.8, 1.0], [0.0, 0.2], [0.8, 0.2]])
        point = np.array([0.0, 0.8])
        expected_below = [False, False, False]

        for idx_end_point, end_point in enumerate(polygon):

            with subtests.test(f"end point {idx_end_point}"):
                start_point = polygon[idx_end_point - 1]
                is_below, distance, vertex_crossing = geo_utils.process_edge(
                    edge_start=start_point, edge_end=end_point, point=point, shift=1e-10
                )
                assert is_below == expected_below[idx_end_point]

        for idx_end_point, end_point in enumerate(polygon):

            with subtests.test(f"end point {idx_end_point}"):
                start_point = polygon[idx_end_point - 1]
                is_below, distance, vertex_crossing = geo_utils.process_edge(
                    edge_start=start_point, edge_end=end_point, point=point, shift=1e-10
                )
                assert vertex_crossing == 0.0

    def test_process_edge_single(self, subtests):

        line = np.array([[0.8, 1.0], [0.0, 1.0]])
        point = np.array([0.8, 0.8])

        is_below, distance, vertex_crossing = geo_utils.process_edge(
            edge_start=line[0], edge_end=line[1], point=point, shift=1e-10
        )
        with subtests.test(f"test below edge"):
            assert is_below == True
        with subtests.test(f"test vertex crossing"):
            assert vertex_crossing == True
        with subtests.test(f"test distance"):
            assert distance == pytest.approx(0.2, rel=1e-7)

    def test_process_edge_single_colinear(self, subtests):

        line = np.array([[0.8, 1.0], [0.0, 1.0]])
        point = np.array([0.4, 1.0])

        is_below, distance, vertex_crossing = geo_utils.process_edge(
            edge_start=line[0], edge_end=line[1], point=point, shift=1e-10
        )
        with subtests.test(f"test below edge"):
            assert is_below == False
        with subtests.test(f"test vertex crossing"):
            assert vertex_crossing == 0
        with subtests.test(f"test distance"):
            assert distance == pytest.approx(0.0, rel=1e-7)

    def test_distance_point_to_unitsquare_center(self):

        point = np.array([0.5, 0.5])
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        expected_distance = -0.5

        test_result = geo_utils.distance_point_to_polygon_ray_casting(
            point, vertices=polygon
        )

        assert test_result == pytest.approx(expected_distance, rel=1e-2)

    def test_distance_point_to_unitsquare_outside(self):

        point = np.array([-0.5, 0.5])
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        expected_distance = 0.5

        test_result = geo_utils.distance_point_to_polygon_ray_casting(
            point, vertices=polygon
        )

        assert test_result == pytest.approx(expected_distance, rel=1e-2)

    def test_distance_point_to_unitsquare_grad_2d(self, subtests):

        point = np.array([-0.25, 0.5], dtype=float)
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        test_result = self.distance_point_to_polygon_ray_casting_grad(
            point, vertices=polygon
        )

        with subtests.test("analytic derivatives"):
            # note that the distance should get more negative as the point moves deeper inside the polygon
            assert np.all(
                test_result[0] == pytest.approx(np.array([-1.0, 0.0], dtype=float))
            )

        with subtests.test("numeric derivatives"):
            try:
                jax.test_util.check_grads(
                    geo_utils.distance_point_to_polygon_ray_casting,
                    args=(point, polygon),
                    order=1,
                    rtol=1e-4,
                )
            except AssertionError:
                pytest.fail(
                    "Unexpected AssertionError when checking gradients, gradients may be incorrect"
                )


@pytest.mark.usefixtures("subtests")
class TestPolygonNormalsCalculator:
    """
    Test for polygon normals calculator
    """

    def test_polygon_normals_calculator_single_polygon(self):

        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        expected_normals = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

        test_result = geo_utils.polygon_normals_calculator(polygon, n_polygons=1)

        assert np.allclose(test_result, expected_normals)

    def test_polygon_normals_calculator_multi_polygon(self, subtests):

        polygon1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = np.array([[5, 0], [6, 0], [6, 1], [5.5, 1], [5, 1]])
        polygons = [polygon1, polygon2]

        expected_normals1 = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        expected_normals2 = np.array([[0, 1], [-1, 0], [0, -1], [0, -1], [1, 0]])
        expected_normals = [expected_normals1, expected_normals2]

        test_result = geo_utils.polygon_normals_calculator(polygons, n_polygons=2)

        for i, r in enumerate(test_result):
            with subtests.test(f"polygon {i}"):
                assert np.allclose(r, expected_normals[i])


@pytest.mark.usefixtures("subtests")
class TestMultiPolygonNormalsCalculator:
    """
    Test for single polygon normals calculator
    """

    def test_multi_polygon_normals_calculator(self):

        polygon1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = np.array([[5, 0], [6, 0], [6, 1], [5, 1]])

        expected_normals = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

        test_result = geo_utils.multi_polygon_normals_calculator(
            np.array([polygon1, polygon2])
        )

        assert np.allclose(test_result, np.array([expected_normals, expected_normals]))

    def test_multi_polygon_normals_calculator_multi_polygon_multi_size(self, subtests):

        polygon1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = np.array([[5, 0], [6, 0], [6, 1], [5.5, 1], [5, 1]])

        expected_normals1 = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        expected_normals2 = np.array([[0, 1], [-1, 0], [0, -1], [0, -1], [1, 0]])
        expected_normals = [expected_normals1, expected_normals2]

        test_result = geo_utils.multi_polygon_normals_calculator([polygon1, polygon2])

        for i, r in enumerate(test_result):
            with subtests.test(f"polygon {i}"):
                assert np.allclose(r, expected_normals[i])


class TestSinglePolygonNormalsCalculator:
    """
    Test for single polygon normals calculator
    """

    def test_single_polygon_normals_calculator(self):
        """
        Test for single polygon normals calculator
        """

        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        expected_normals = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

        test_result = geo_utils.single_polygon_normals_calculator(polygon)

        assert np.allclose(test_result, expected_normals)

    def test_single_polygon_normals_calculator_rotated(self):
        """
        Test for single polygon normals calculator
        """

        polygon = np.array([[0, 0], [1, 1], [0, 2], [-1, 1]])
        expected_normals = np.array(
            [
                [-0.7071067811865476, 0.7071067811865476],
                [-0.7071067811865476, -0.7071067811865476],
                [0.7071067811865476, -0.7071067811865476],
                [0.7071067811865476, 0.7071067811865476],
            ]
        )

        test_result = geo_utils.single_polygon_normals_calculator(polygon)

        assert np.allclose(test_result, expected_normals)

    def test_single_polygon_normals_calculator_rotated_concave(self):
        """
        Test for single polygon normals calculator
        """

        polygon = np.array(
            [[0, 0], [1, 1], [0.5, 1.5], [0.0, 1.0], [-0.5, 1.5], [-1, 1]]
        )
        expected_normals = np.array(
            [
                [-0.7071067811865476, 0.7071067811865476],
                [-0.7071067811865476, -0.7071067811865476],
                [0.7071067811865476, -0.7071067811865476],
                [-0.7071067811865476, -0.7071067811865476],
                [0.7071067811865476, -0.7071067811865476],
                [0.7071067811865476, 0.7071067811865476],
            ]
        )

        test_result = geo_utils.single_polygon_normals_calculator(polygon)

        assert np.allclose(test_result, expected_normals)


class TestPointOnLine:
    """
    Test for point on line segment
    """

    def test_point_on_line_middle_not_on_line(self):

        point = np.array([10, 10])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.point_on_line(point, line_a, line_b)

        assert test_result == False

    def test_point_on_line_middle(self):

        point = np.array([5, 5])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.point_on_line(point, line_a, line_b)

        assert test_result == True

    def test_point_on_line_end(self):

        point = np.array([10, 0])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.point_on_line(point, line_a, line_b)

        assert test_result == True

    def test_point_on_line_end_not_on_line(self):

        point = np.array([-1, 11])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.point_on_line(point, line_a, line_b)

        assert test_result == False

    def test_point_on_line_middle_lt_tol(self):

        point = np.array([5 + 1e-7, 5 + 1e-7])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.point_on_line(point, line_a, line_b)

        assert test_result == True

    def test_point_on_line_middle_gt_tol(self):

        point = np.array([5 + 1e-6, 5 + 1e-6])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.point_on_line(point, line_a, line_b)

        assert test_result == False


@pytest.mark.usefixtures("subtests")
class TestGetClosestPointOnLineSeg:
    def setup_method(self):
        self.get_closest_point_on_line_seg_jac = jax.jacobian(
            geo_utils.get_closest_point_on_line_seg, [0]
        )
        pass

    def test_get_closest_point_on_line_seg_45_deg_with_end(self):

        point = np.array([10, 10])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point_on_line_seg(
            point, line_a, line_b, line_vector
        )

        assert np.all(test_result == np.array([5, 5]))

    def test_get_closest_point_on_line_seg_90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([0, 10])
        line_b = np.array([10, 10])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point_on_line_seg(
            point, line_a, line_b, line_vector
        )

        assert np.all(test_result == line_a)

    def test_get_closest_point_on_line_seg_gt90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point_on_line_seg(
            point, line_a, line_b, line_vector
        )

        assert np.all(test_result == line_a)

    def test_get_closest_point_on_line_seg_180_deg_with_end(self):

        point = np.array([0, 5])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point_on_line_seg(
            point, line_a, line_b, line_vector
        )

        assert np.all(test_result == line_a)

    def test_get_closest_point_on_line_seg_point_on_segment(self):
        """
        Test for a point exactly on the line segment
        """

        test_point = np.array([3, 3, 3])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point_on_line_seg(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == test_point)

    def test_get_closest_point_on_line_seg_point_near_end(self):
        """
        Test for a point near the end of the line segment
        """

        test_point = np.array([6, 6, 6])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point_on_line_seg(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == test_end)

    def test_get_closest_point_on_line_seg_point_near_start(self):
        """
        Test for a point near the start of the line segment
        """

        test_point = np.array([-1, -1, -2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point_on_line_seg(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == test_start)

    def test_get_closest_point_on_line_seg_point_near_middle(self):
        """
        Test for a point near the middle of the line segment
        """

        test_point = np.array([5, 5, 2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([0, 0, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point_on_line_seg(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == np.array([0, 0, 2]))

    def test_get_closest_point_on_line_seg_jac(self, subtests):
        """
        Test for gradient for a point near the middle of the line segment
        """

        test_point = np.array([5, 0, 2], dtype=float)
        test_start = np.array([0, 0, 0], dtype=float)
        test_end = np.array([0, 0, 5], dtype=float)
        line_vector = test_end - test_start

        tr_dp = self.get_closest_point_on_line_seg_jac(
            test_point, test_start, test_end, line_vector
        )

        with subtests.test("analytic derivatives"):
            assert np.all(
                tr_dp == np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)
            )

        with subtests.test("numeric derivatives"):
            try:
                jax.test_util.check_grads(
                    geo_utils.get_closest_point_on_line_seg,
                    (test_point, test_start, test_end, line_vector),
                    order=1,
                )
            except AssertionError:
                pytest.fail(
                    "Unexpected AssertionError when checking gradients, gradients may be incorrect"
                )


class TestPointToLineSeg:
    def setup_method(self):
        self.distance_point_to_lineseg_nd_grad = jax.grad(
            geo_utils.distance_point_to_lineseg_nd, [0]
        )
        pass

    def test_distance_point_to_lineseg_nd_45_deg_with_end(self):

        point = np.array([10, 10])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 7.0710678118654755

    def test_distance_point_to_lineseg_90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([0, 10])
        line_b = np.array([10, 10])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 10.0

    def test_distance_point_to_lineseg_gt90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 7.0710678118654755

    def test_distance_point_to_lineseg_180_deg_with_end(self):

        point = np.array([0, 5])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 5.0

    def test_distance_point_to_lineseg_nd_point_to_point(self):
        """
        Test for a point to point calculation
        """

        test_point = np.array([5, 5, 5])
        test_start = np.array([0, 0, 0])
        test_end = np.array([0, 0, 0])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(8.660254037844387)

    def test_distance_point_to_lineseg_nd_point_on_segment(self):
        """
        Test for a point exactly on the line segment
        """

        test_point = np.array([3, 3, 3])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(0.0)

    def test_distance_point_to_lineseg_nd_point_near_end(self):
        """
        Test for a point near the end of the line segment
        """

        test_point = np.array([6, 6, 6])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(1.7320508075688772)

    def test_distance_point_to_lineseg_nd_point_near_start(self):
        """
        Test for a point near the start of the line segment
        """

        test_point = np.array([-1, -1, -2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(2.449489742783178)

    def test_distance_point_to_lineseg_nd_point_near_middle(self):
        """
        Test for a point near the middle of the line segment
        """

        test_point = np.array([5, 5, 2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([0, 0, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(7.0710678118654755)

    def test_distance_point_to_lineseg_nd_grad(self):
        """
        Test for gradient for a point near the middle of the line segment
        """

        test_point = np.array([5, 0, 2], dtype=float)
        test_start = np.array([0, 0, 0], dtype=float)
        test_end = np.array([0, 0, 5], dtype=float)

        tr_dp = self.distance_point_to_lineseg_nd_grad(test_point, test_start, test_end)

        assert np.all(tr_dp == np.array([1, 0, 0]))

        try:
            jax.test_util.check_grads(
                geo_utils.distance_point_to_lineseg_nd,
                (test_point, test_start, test_end),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestLineSegToLineSeg:
    def setup_method(self):
        self.distance_lineseg_to_lineseg_nd_grad = jax.grad(
            geo_utils.distance_lineseg_to_lineseg_nd, [0]
        )
        pass

    def test_distance_lineseg_to_lineseg_nd_parallel_2d(self):
        """
        Test distance between line segments 2d for parallel lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])])
        line_b = np.array([np.array([5, 0]), np.array([5, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(
            5.0, rel=1e-1
        )  # inexact with parallel lines due to smooth max/min

    def test_distance_lineseg_to_lineseg_nd_shared_point_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])])
        line_b = np.array([np.array([0, 0]), np.array([5, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect_2d(self):
        """
        Test distance between line segments 2d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0]), np.array([5, 5])])
        line_b = np.array([np.array([0, 5]), np.array([5, 0])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5]), np.array([5, 5])])
        line_b = np.array([np.array([0, 5]), np.array([5, 0])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_skew_2d(self):
        """
        Test distance between line segments 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])])
        line_b = np.array([np.array([5, 5]), np.array([8, -10])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0, rel=1e-3)

    def test_distance_lineseg_to_lineseg_nd_skew_2d_2(self):
        """
        Test gradient of the distance between line segments in 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([5.0, 2.5]), np.array([10, 15])], dtype=float)

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0)

    def test_distance_lineseg_to_lineseg_nd_skew_2d_3(self):
        """
        Test gradient of the distance between line segments in 2d for skew lines
        """

        line_a = np.array([np.array([5.0, 2.5]), np.array([10, 15])], dtype=float)
        line_b = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0)

    def test_distance_lineseg_to_lineseg_nd_parallel_grad_2d(self):
        """
        Test grad of distance between line segments 2d for parallel lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([5, 0]), np.array([0, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(
            test_result == np.array([0.0, 0.0])
        )  # goes to zero due to smooth_norm

    def test_distance_lineseg_to_lineseg_nd_shared_point_grad_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([0, 0]), np.array([5, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_grad_2d(self):
        """
        Test distance between line segments 2d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0]), np.array([5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5]), np.array([5, 0])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line_grad_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5]), np.array([5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5]), np.array([5, 0])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_skew_grad_2d(self):
        """
        Test gradient of the distance between line segments in 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([5.0, 2.5]), np.array([10, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(
            test_result[0] == pytest.approx(np.array([-0.5, 0.0], dtype=float))
        )

        try:
            jax.test_util.check_grads(
                geo_utils.distance_lineseg_to_lineseg_nd,
                (line_a[0], line_a[1], line_b[0], line_b[1]),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )

    def test_distance_lineseg_to_lineseg_nd_2d_skew_grad(self):
        """
        Test distance between line segments 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 10])], dtype=float)
        line_b = np.array([np.array([5, 0]), np.array([6, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([-1, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_2d_skew_grad(self):
        """
        Test distance between line segments 2d for skew lines
        """

        line_b = np.array([np.array([0, 0]), np.array([0, 10])], dtype=float)
        line_a = np.array([np.array([5, 0]), np.array([6, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([1, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_parallel(self):
        """
        Test distance between line segments 3d for parallel lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])])
        line_b = np.array([np.array([5, 0, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(
            5.0, rel=1e-1
        )  # inexact with parallel lines due to smooth max/min

    def test_distance_lineseg_to_lineseg_nd_shared_point(self):
        """
        Test distance between line segments 3d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])])
        line_b = np.array([np.array([0, 0, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect(self):
        """
        Test distance between line segments 3d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([5, 5, 5])])
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line(self):
        """
        Test distance between line segments 3d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5, 2.5]), np.array([5, 5, 5])])
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_skew(self):
        """
        Test distance between line segments 3d for skew lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 5, 5])])
        line_b = np.array([np.array([5, 5, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0, rel=1e-3)

    def test_distance_lineseg_to_lineseg_nd_parallel_grad(self):
        """
        Test grad of distance between line segments 3d for parallel lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])], dtype=float)
        line_b = np.array([np.array([5, 0, 0]), np.array([5, 0, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        # the grad is prone to error in this region due to smooth max with many similar values,
        # just make sure grad is correct sign and reasonable magnitude.
        # Exact value should be [-1.0, 0.0, 0.0]
        assert np.all(test_result <= np.array([-0.5, 0.0, 0.0]))

    def test_distance_lineseg_to_lineseg_nd_almost_parallel_grad(self):
        """
        Test grad of distance between line segments 3d for parallel lines
        """

        line_a = np.array([np.array([0.01, 0, 0]), np.array([0, 0, 5])], dtype=float)
        line_b = np.array([np.array([5, 0, 0]), np.array([5, 0, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([-1.0, 0.0, 0.0]))

    def test_distance_lineseg_to_lineseg_nd_shared_point_grad(self):
        """
        Test distance between line segments 3d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])], dtype=float)
        line_b = np.array([np.array([0, 0, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_grad(self):
        """
        Test distance between line segments 3d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([5, 5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line_grad(self):
        """
        Test distance between line segments 3d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5, 2.5]), np.array([5, 5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_skew_grad(self):
        """
        Test distance between line segments 3d for skew lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 5, 5])], dtype=float)
        line_b = np.array([np.array([5, 5, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([-0.5, 0, 0], dtype=float))

        try:
            jax.test_util.check_grads(
                geo_utils.distance_lineseg_to_lineseg_nd,
                (line_a[0], line_a[1], line_b[0], line_b[1]),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )
