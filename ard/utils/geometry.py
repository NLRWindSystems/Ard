import numpy as np
import jax.numpy as jnp
import jax
from ard.utils.mathematics import smooth_max, smooth_min, smooth_norm


def get_nearest_polygons(
    boundary_vertices,
    points_x,
    points_y,
    s=700,
    tol=1e-6,
):
    """
    Determines the nearest polygon for each point using the ray-casting algorithm. This
    function may be used to assign turbines to regions in a wind farm layout, but is not
    intended for use in a gradient-based optimization context. The function is not
    differentiable. This implementation is based on FLOWFarm.jl
    (https://github.com/byuflowlab/FLOWFarm.jl)

    Args:
        boundary_vertices (np.ndarray or list of np.ndarray): Vertices of the boundary in
            counterclockwise order. If `discrete` is True, this should be a list of arrays.
        points_x (np.ndarray): points x coordinates.
        points_y (np.ndarray): points y coordinates.
        s (float, optional): Smoothing factor for smooth max. Defaults to 700.
        tol (float, optional): Tolerance for determining proximity. Defaults to 1e-6.

    Returns:
        np.ndarray: Region assignments for each turbine.
    """

    # Number of turbines
    n_points = len(points_x)

    # Number of regions
    nregions = len(boundary_vertices)

    # Initialize region and status arrays
    region = -np.ones(n_points, dtype=int)

    # Initialize array to hold distances from each turbine to the closest boundary face
    turbine_to_region_distance = np.zeros(nregions, dtype=float)

    for i in range(n_points):
        # Iterate through each region
        for k in range(nregions):
            # Get distance to each region
            ctmp = distance_point_to_polygon_ray_casting(
                np.array([points_x[i], points_y[i]]),
                boundary_vertices[k],
                s=s,
                shift=tol,
                return_distance=True,
            )
            if ctmp <= 0:  # Negative if in boundary
                region[i] = k
                break

            # Check if the point is not in any of the regions
            else:
                # Calculate distance to each region
                turbine_to_region_distance[k] = ctmp

        # Magnitude of the constraint value
        if region[i] == -1:
            # Indicate closest region
            region[i] = np.argmin(turbine_to_region_distance)

    return region

# Pad all polygons to have the same number of vertices
def pad_polygon(polygon, max_vertices):
    padding = max_vertices - len(polygon)
    return jnp.pad(polygon, ((0, padding), (0, 0)), mode="edge")

def distance_multi_point_to_multi_polygon_ray_casting(
    points_x: np.ndarray[float],
    points_y: np.ndarray[float],
    boundary_vertices: list[list[np.ndarray]],
    regions: np.ndarray[int],
    s=700,
    tol=1e-6,
) -> np.ndarray:
    """
    Calculate the distance from each point to the nearest point on a polygon or set of polygons using
    the ray-casting (Jordan curve theorem) algorithm. Negative means the turbine is inside at least
    one polygon. This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl)

    Args:
        points_x (np.ndarray[list]): points x coordinates.
        points_y (np.ndarray[list]): points y coordinates.
        boundary_vertices (list[list[np.ndarray]]): Vertices of the each boundary in
            counterclockwise order. Boundaries should be simple polygons but do not need to have the same number of vertices.
        regions (np.array[int]): Predefined region assignments for each point. Defaults to None.
        s (float, optional): Smoothing factor for smooth max. Defaults to 700.
        tol (float, optional): Tolerance for determining proximity of point to polygon to be considered inside the polygon. Defaults to 1e-6.
        c (np.ndarray, optional): Preallocated array for constraint values. Defaults to None.

    Returns:
        np.ndarray: Constraint values for each turbine.
        np.ndarray (optional): Region assignments for each turbine (if `return_region` is True).
    """
    # Combine points_x and points_y into a single array of points
    points = jnp.stack([points_x, points_y], axis=1)

    # Convert boundary_vertices to JAX arrays
    boundary_vertices_jax = [jnp.asarray(poly, dtype=jnp.float32) for poly in boundary_vertices]
    
    # Create a function for each polygon that computes distance
    def make_distance_func(vertices):
        def compute_for_polygon(point):
            return distance_point_to_polygon_ray_casting(
                point=point,
                vertices=vertices,
                s=s,
                shift=tol,
                return_distance=True,
            )
        return compute_for_polygon
    
    # Create list of functions, one per polygon
    distance_funcs = [make_distance_func(vertices) for vertices in boundary_vertices_jax]
    
    # Define function to compute distance using jax.lax.switch
    def compute_distance(point, region_idx):
        return jax.lax.switch(region_idx, distance_funcs, point)
    
    # Vectorize over all points
    distances = jax.vmap(compute_distance)(points, regions)

    return distances


distance_multi_point_to_multi_polygon_ray_casting = jax.jit(
    distance_multi_point_to_multi_polygon_ray_casting
)

# Define a function to process a single edge
def process_edge(
    edge_start: jnp.ndarray, 
    edge_end: jnp.ndarray, 
    point: jnp.ndarray, 
    shift: float,
):
    """
    Process a single polygon edge for ray-casting algorithm.
    
    Determines if a vertical ray cast from the point up intersects
    the edge, and calculates the distance from the point to the edge segment.
    
    Args:
        edge_start (jnp.ndarray): Start vertex of the edge ([x, y]).
        edge_end (jnp.ndarray): End vertex of the edge ([x, y]).
        point (jnp.ndarray): Point of interest ([x, y]).
        shift (float): Small value added to avoid division by zero when edge is vertical.
    
    Returns:
        tuple: A tuple containing:
            - is_below (bool): True if point is below the edge at the ray intersection.
            - distance (float): Shortest distance from point to the edge segment.
            - vertex_crossing (int): 1 if ray crosses exactly at edge_start vertex, 0 otherwise.
    """

    # Check if the x-coordinate of the point is between the x-coordinates of the edge, 
    # including when the point is directly below a vertical edge
    x_condition = ((edge_start[0] <= point[0]) & (point[0] < edge_end[0])) | (
                   (edge_start[0] >= point[0]) & (point[0] > edge_end[0])) | (
                   (jnp.isclose(edge_start[0], edge_end[0]) & jnp.isclose(point[0], edge_start[0]))
                   )

    # Calculate the y-coordinate of the edge at the x-coordinate of the point
    y = ((edge_end[1] - edge_start[1]) / (edge_end[0] - edge_start[0] + shift)) * (
        point[0] - edge_start[0] + shift
    ) + edge_start[1]

    # Determine if the point is below the edge
    is_below = x_condition & (point[1] < y)

    # record vertex crossings
    vertex_crossing = jnp.where(
        x_condition & (edge_start[0] == point[0]) & is_below,
        1,
        0
    )

    # Calculate the distance to the edge
    distance = distance_point_to_lineseg_nd(point, edge_start, edge_end)

    return is_below, distance, vertex_crossing

# Vectorize the edge processing function
process_edge_vec = jax.vmap(process_edge, in_axes=(0, 0, None, None))

def distance_point_to_polygon_ray_casting(
    point: jnp.ndarray,
    vertices: jnp.ndarray,
    s: float = 700,
    shift: float = 1e-10,
    return_distance: bool = True,
):
    """
    Determines the signed distance from a point to a polygon using the Jordan curve
    theorem (ray-casting) approach as discussed in [1] and [2]. The polygon is
    assumed to be simple and defined in counterclockwise order. Complex polygons
    (where edges cross one another) are not supported. The function is
    differentiable with respect to the point coordinates.

    [1] Numerical Recipes: The Art of Scientific Computing by Press, et al. 3rd edition, sec. 21.4.3 (p. 1124)
    [2] https://en.wikipedia.org/wiki/Point_in_polygon

    Args:
        point (jnp.ndarray): Point of interest (2D vector).
        vertices (jnp.ndarray): Vertices of the polygon (Nx2 array) in counterclockwise order.
        s (float, optional): Smoothing factor for the smoothmin function. Defaults to 700.
        shift (float, optional): Small shift to handle edge cases. Defaults to 1e-10.
        return_distance (bool, optional): Whether to return the signed distance or just
            inside/outside status. Defaults to True. When False, the function is not
            differentiable.

    Returns:
        float: Signed distance or inside/outside status. Negative if inside, positive if outside.
    """
    
    # Ensure inputs are JAX arrays with explicit data types
    point = jnp.asarray(point, dtype=jnp.float32)
    vertices = jnp.asarray(vertices, dtype=jnp.float32)

    # Add the first vertex to the end to close the polygon loop
    vertices = jnp.vstack([vertices, vertices[0]])

    edge_starts = vertices[:-1]
    edge_ends = vertices[1:]
    is_below, distances, vertex_crossings = process_edge_vec(edge_starts, edge_ends, point, shift)

    # Check for tangential vertex crossings
    # Create arrays of current and previous edge differences
    curr_edge_end_diff = edge_ends[:, 0] - point[0]
    prev_edge_start_diff = jnp.roll(edge_starts[:, 0], 1) - point[0]  # Shift by 1 to get previous
    # Tangential crossing occurs when vertex_crossing > 0 AND the other two connected vertices are on same side
    tangential_vertex_crossings = jnp.where(
        (vertex_crossings > 0) & (curr_edge_end_diff * prev_edge_start_diff > 0),
        1,
        0
    )

    # Count the number of intersections, ignoring tangential vertex crossings
    intersection_counter = jnp.sum(is_below) - jnp.sum(tangential_vertex_crossings)

    # Compute the signed distance if desired
    if return_distance:
        c_prime = smooth_min(distances, s=s)
    else:
        c_prime = 1

    c = jax.lax.cond(
            intersection_counter % 2 == 1,
            lambda _: -c_prime,
            lambda _: c_prime,
            operand=None,
        )
    
    return c


def polygon_normals_calculator(
    boundary_vertices: np.ndarray, n_polygons: int = 1
) -> list[np.ndarray]:
    """
    Calculate unit vectors perpendicular to each edge of each polygon in a set of polygons.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).
    This function is not intended to be differentiable wrt n_polygons.

    Args:
        boundary_vertices (list of np.ndarray): List of m-by-2 arrays, where each array contains
            the boundary vertices of a polygon in counter-clockwise order. The number of vertices (m)
            in each polygon does not need to be the same.
        nboundaries (int, optional): The number of boundaries in the set. Defaults to 1.

    Returns:
        list of np.ndarray: List of m-by-2 arrays of unit vectors perpendicular to each edge
            of each polygon.
    """
    if n_polygons == 1:
        # Single polygon case
        return single_polygon_normals_calculator(boundary_vertices)
    else:
        # Multiple polygons case
        return multi_polygon_normals_calculator(boundary_vertices)


def multi_polygon_normals_calculator(
    boundary_vertices: list[np.ndarray],
) -> jnp.ndarray:
    """
    Calculate unit vectors perpendicular to each edge of each polygon in a set of polygons.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).

    Args:
        boundary_vertices (list of np.ndarray): List of m-by-2 arrays, where each array contains
            the boundary vertices of a polygon in counterclockwise order.
        nboundaries (int, optional): The number of boundaries in the set. Defaults to 1.

    Returns:
        list of np.ndarray: List of m-by-2 arrays of unit vectors perpendicular to each edge
            of each polygon.
    """
    return [
        single_polygon_normals_calculator(vertices) for vertices in boundary_vertices
    ]


multi_polygon_normals_calculator = jax.jit(multi_polygon_normals_calculator)


def single_polygon_normals_calculator(boundary_vertices: np.ndarray) -> jnp.ndarray:
    """
    Calculate unit vectors perpendicular to each edge of a polygon pointing into the polygon.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).

    Args:
        boundary_vertices (np.ndarray): m-by-2 array containing all the boundary vertices
            in counterclockwise order.

    Returns:
        np.ndarray: m-by-2 array of unit vectors perpendicular to each edge of the polygon pointing into the polygon.
    """

    # Add the first vertex to the end to form a closed loop
    boundary_vertices = jnp.vstack([boundary_vertices, boundary_vertices[0]])

    # Compute the differences (dx, dy) for each edge
    dx = boundary_vertices[1:, 0] - boundary_vertices[:-1, 0]
    dy = boundary_vertices[1:, 1] - boundary_vertices[:-1, 1]

    # Create vectors normal to the boundary edges
    boundary_normals = jnp.stack([-dy, dx], axis=1)

    # Normalize the vectors
    norms = jnp.linalg.norm(boundary_normals, axis=1, keepdims=True)
    boundary_normals = boundary_normals / norms

    return [boundary_normals]


single_polygon_normals_calculator = jax.jit(single_polygon_normals_calculator)


def point_on_line(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, tol=1e-6):
    """
    Determine if a point lies on a line segment.

    Given a line determined by two points (v1 and v2), determine if the point (p) lies on the line
    between those points within a given tolerance.

    Args:
        p (np.ndarray): Point of interest (2D vector).
        v1 (np.ndarray): First vertex of the line (2D vector).
        v2 (np.ndarray): Second vertex of the line (2D vector).
        tol (float): Tolerance for determining co-linearity.

    Returns:
        bool: True if the point lies on the line, False otherwise.
    """

    d = distance_point_to_lineseg_nd(p, v1, v2)

    return jnp.isclose(d, 0.0, atol=tol)


def _distance_lineseg_to_lineseg_coplanar(
    line_a_start: np.ndarray,
    line_a_end: np.ndarray,
    line_b_start: np.ndarray,
    line_b_end: np.ndarray,
) -> float:
    """Returns the distance between two finite line segments assuming the segments are coplanar.
    It is up to the user to check the required condition. There may be some error in the case
    when the line segments are parallel since multiple points may have equal distances, leading to
    some error from the smooth minimum function.

    Args:
        line_a_start (np.ndarray): start point of line a
        line_a_end (np.ndarray): end point of line a
        line_b_start (np.ndarray): start point of line b
        line_b_end (np.ndarray): end point of line b

    Returns:
        distance (float): the distance between the lines
    """

    # get distance between all pairs of end points
    a_start_to_b = distance_point_to_lineseg_nd(line_a_start, line_b_start, line_b_end)
    a_end_to_b = distance_point_to_lineseg_nd(line_a_end, line_b_start, line_b_end)
    b_start_to_a = distance_point_to_lineseg_nd(line_b_start, line_a_start, line_a_end)
    b_end_to_a = distance_point_to_lineseg_nd(line_b_end, line_a_start, line_a_end)

    distance = smooth_min(
        jnp.array([a_start_to_b, a_end_to_b, b_start_to_a, b_end_to_a])
    )

    return distance


_distance_lineseg_to_lineseg_coplanar = jax.jit(_distance_lineseg_to_lineseg_coplanar)


def distance_lineseg_to_lineseg_nd(
    line_a_start: np.ndarray,
    line_a_end: np.ndarray,
    line_b_start: np.ndarray,
    line_b_end: np.ndarray,
    tol=1e-12,
) -> float:
    """Find the distance between two line segments in 2d or 3d. This method is primarily based on reference [1],
    using a parametric approach based on the determinant and cross product to find the closest points on the two line
    segments. However, to handle the special case of line segments that are coplanar, we use the smooth minimum of the
    distance between the endpoints of the two line segments and the other line segment. In the coplanar case, the
    returned distance between the two line segments may have a noticeable error due to possibly having multiple points
    with the same distance, which leads to error in the smooth minimum function.

    [1] Numerical Recipes: The Art of Scientific Computing by Press, et al. 3rd edition, sec. 21.4.2 (p. 1121)

    Args:
        line_a_start (np.ndarray): The start point of line segment "a" as either [x,y,z] or [x,y]
        line_a_end (np.ndarray): The end point of line segment "a" as either [x,y,z] or [x,y]
        line_b_start (np.ndarray): The start point of line segment "b" as either [x,y,z] or [x,y]
        line_b_end (np.ndarray): The end point of line segment "b" as either [x,y,z] or [x,y]
        tol (float, optional): If denominator in key equation is less than or equal tol, then an alternative method is used. Defaults to 0.0.

    Returns:
        float: Distance between the two line segments
    """

    def a_is_point(inputs0i) -> float:
        line_a_start = inputs0i[0]
        line_b_start = inputs0i[2]
        line_b_end = inputs0i[3]
        return distance_point_to_lineseg_nd(line_a_start, line_b_start, line_b_end)

    def b_is_point(inputs0i) -> float:
        line_a_start = inputs0i[0]
        line_a_end = inputs0i[1]
        line_b_start = inputs0i[2]
        return distance_point_to_lineseg_nd(line_b_start, line_a_start, line_a_end)

    def a_is_not_point(inputs0i) -> float:
        line_b_vector = inputs0i[5]
        return jax.lax.cond(
            jnp.all(line_b_vector == 0.0), b_is_point, a_and_b_are_lines, inputs0i
        )

    def a_and_b_are_lines(inputs0i) -> float:

        def denom_lt_tol(inputs1i) -> float:
            line_a_start = inputs1i[0]
            line_a_end = inputs1i[1]
            line_b_start = inputs1i[2]
            line_b_end = inputs1i[3]
            return _distance_lineseg_to_lineseg_coplanar(
                line_a_start=line_a_start,
                line_a_end=line_a_end,
                line_b_start=line_b_start,
                line_b_end=line_b_end,
            )

        def denom_gt_tol(inputs1i) -> float:
            line_a_start = inputs1i[0]
            line_a_end = inputs1i[1]
            line_b_start = inputs1i[2]
            line_b_end = inputs1i[3]
            line_a_vector = inputs1i[4]
            line_b_vector = inputs1i[5]
            denominator = inputs1i[6]

            a = line_a_start
            v = line_a_vector
            x = line_b_start
            u = line_b_vector

            s_numerator = jnp.linalg.det(jnp.array([a - x, u, jnp.cross(u, v)]).T)
            t_numerator = jnp.linalg.det(jnp.array([a - x, v, jnp.cross(u, v)]).T)

            s = s_numerator / denominator
            t = t_numerator / denominator

            # Get closest point along the lines
            # if s or t > 1, use end point of line
            def st_gt_1(inputs23i) -> np.ndarray:
                line_end = inputs23i[1]
                return jnp.array(line_end, dtype=float)

            # if s or t < 0, use start point of line
            def st_lt_0(inputs23i) -> np.ndarray:
                line_start = inputs23i[0]
                return jnp.array(line_start, dtype=float)

            # otherwise compute the closest point on line using the parametric form of the line segment
            def st_gt_0_lt_1(inputs23i) -> np.ndarray:
                line_start = inputs23i[0]
                line_vector = inputs23i[2]
                st = inputs23i[3]
                return jnp.array(line_start + st * line_vector, dtype=float)

            def st_lt_1(inputs23i) -> np.ndarray:
                st = inputs23i[3]
                return jax.lax.cond(st < 0, st_lt_0, st_gt_0_lt_1, inputs23i)

            # get closest point on lines a and b to each other
            inputs2o = [line_a_start, line_a_end, line_a_vector, s]
            closest_point_line_a = jax.lax.cond(s > 1, st_gt_1, st_lt_1, inputs2o)
            inputs3o = [line_b_start, line_b_end, line_b_vector, t]
            closest_point_line_b = jax.lax.cond(t > 1, st_gt_1, st_lt_1, inputs3o)

            # the distance between the line segments is the distance between the closest points (in many cases)
            parametric_distance = smooth_norm(
                closest_point_line_b - closest_point_line_a
            )

            # parametric approach can miss cases, so compare with point to line distances
            distance_point_a_line_b = distance_point_to_lineseg_nd(
                closest_point_line_a, line_b_start, line_b_end
            )
            distance_point_b_line_a = distance_point_to_lineseg_nd(
                closest_point_line_b, line_a_start, line_a_end
            )
            distance = smooth_min(
                jnp.array(
                    [
                        parametric_distance,
                        distance_point_a_line_b,
                        distance_point_b_line_a,
                    ]
                )
            )

            return distance

        line_a_start = inputs0i[0]
        line_a_end = inputs0i[1]
        line_b_start = inputs0i[2]
        line_b_end = inputs0i[3]
        line_a_vector = inputs0i[4]
        line_b_vector = inputs0i[5]

        # find s and t (point along segment where the segments are closest to each other) using eq. 21.4.17 in [1]
        denominator = smooth_norm(jnp.cross(line_b_vector, line_a_vector)) ** 2

        inputs1o = [
            line_a_start,
            line_a_end,
            line_b_start,
            line_b_end,
            line_a_vector,
            line_b_vector,
            denominator,
        ]

        distance = jax.lax.cond(
            denominator <= tol, denom_lt_tol, denom_gt_tol, inputs1o
        )

        return distance

    # if 2d given, then pad with zeros to get 3d points
    pad_width = len(line_a_start)
    line_a_start = jnp.pad(line_a_start, (0, 3 - pad_width))
    line_a_end = jnp.pad(line_a_end, (0, 3 - pad_width))
    line_b_start = jnp.pad(line_b_start, (0, 3 - pad_width))
    line_b_end = jnp.pad(line_b_end, (0, 3 - pad_width))

    line_a_vector = line_a_end - line_a_start
    line_b_vector = line_b_end - line_b_start

    inputs0o = [
        line_a_start,
        line_a_end,
        line_b_start,
        line_b_end,
        line_a_vector,
        line_b_vector,
    ]

    distance = jax.lax.cond(
        jnp.all(line_a_vector == 0.0), a_is_point, a_is_not_point, inputs0o
    )

    return distance


distance_lineseg_to_lineseg_nd = jax.jit(distance_lineseg_to_lineseg_nd)


def distance_point_to_lineseg_nd(
    point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
) -> float:
    """Find the distance from a point to a line segment in N-Dimensions. This
    implementation can handle any number of dimensions as well as the reduced case
    of point to point distance. If the same point is passed for the start and end
    of the line segment, then the distance is simply the distance from the point
    of interest to the single start/end point.

    Args:
        point (np.ndarray): point of interest [x,y,...]
        segment_start (np.ndarray): start point of line segment [x,y,...]
        segment_end (np.ndarray): end point of line segment [x,y,...]

    Returns:
        distance (float): shortest distance between the point and finite line segment
    """

    def if_point_to_point(inputs) -> float:
        point = inputs[0]
        segment_start = inputs[1]
        return jnp.float64(smooth_norm(segment_start - point))

    def if_point_to_line_seg(inputs) -> float:
        point = inputs[0]
        segment_start = inputs[1]
        segment_end = inputs[2]
        segment_vector = inputs[3]

        # get the closest point on the line segment to the point of interest
        closest_point = get_closest_point_on_line_seg(
            point, segment_start, segment_end, segment_vector
        )

        # the distance from the point to the line is the distance from the point to the closest point on the line
        return jnp.float64(smooth_norm(point - closest_point))

    # get the vector of the line segment
    segment_vector = segment_end - segment_start

    # if the segment is a point, then get the distance to that point
    distance = jax.lax.cond(
        jnp.all(segment_vector == 0),
        if_point_to_point,
        if_point_to_line_seg,
        [point, segment_start, segment_end, segment_vector],
    )

    return distance


distance_point_to_lineseg_nd = jax.jit(distance_point_to_lineseg_nd)


def get_closest_point_on_line_seg(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    segment_vector: np.ndarray,
) -> np.ndarray:
    """Get the closest point on a line segment to the point of interest in N-Dimensions
    using vector projection.

    Args:
        point (np.ndarray): point of interest [x,y,...]
        segment_start (np.ndarray): start point of line segment [x,y,...]
        segment_end (np.ndarray): end point of line segment [x,y,...]
        segment_vector (np.ndarray): segment_end - segment_start

    Returns:
        np.ndarray: closest point on the line segment to the point of interest
    """

    # calculate the distance to the starting point
    start_to_point_vector = point - segment_start

    # calculate the unit vector projection of the start to point vector on the line segment
    projection = jnp.divide(
        jnp.dot(start_to_point_vector, segment_vector),
        jnp.dot(segment_vector, segment_vector),
    )

    def lt_0(inputs) -> np.ndarray:
        segment_start = inputs[1]
        return jnp.array(segment_start, dtype=jnp.float64)

    def gt_1(inputs) -> np.ndarray:
        segment_end = inputs[2]
        return jnp.array(segment_end, dtype=jnp.float64)

    def gt_0(inputs) -> np.ndarray:
        projection = inputs[0]
        return jax.lax.cond(projection > 1, gt_1, lt_1_gt_0, inputs)

    def lt_1_gt_0(inputs) -> np.ndarray:
        projection = inputs[0]
        segment_start = inputs[1]
        segment_vector = inputs[3]
        return jnp.array(segment_start + projection * segment_vector, dtype=jnp.float64)

    return jax.lax.cond(
        projection < 0,
        lt_0,
        gt_0,
        [projection, segment_start, segment_end, segment_vector],
    )


get_closest_point_on_line_seg = jax.jit(get_closest_point_on_line_seg)
