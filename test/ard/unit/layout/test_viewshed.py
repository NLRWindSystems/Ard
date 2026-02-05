import pytest
import numpy as np
from ard.layout import viewshed

## HELPER FUNCTION TESTING


# make sure that for various values the viewshed angles are in a sensible range
@pytest.mark.parametrize(
    "D_rotor, h_hub, h_terrain, expected_range",
    [
        (100.0, 80.0, 0.0, (0.0, 1.0 * (2.0 * np.pi / 360.0))),
        (120.0, 100.0, 10.0, (0.0, 1.0 * (2.0 * np.pi / 360.0))),
        (50.0, 60.0, 5.0, (0.0, 1.0 * (2.0 * np.pi / 360.0))),
    ],
)
def test_calculate_viewshed_section_angle(D_rotor, h_hub, h_terrain, expected_range):
    angle = viewshed.calculate_viewshed_section_angle(
        D_rotor, h_hub, h_terrain=h_terrain
    )
    assert expected_range[0] <= angle <= expected_range[1]


# make sure the small-angle approximation is coming in under the real number
@pytest.mark.parametrize(
    "D_rotor, h_hub, h_terrain",
    [
        (100.0, 80.0, 0.0),
        (120.0, 100.0, 10.0),
        (50.0, 60.0, 5.0),
    ],
)
def test_calculate_viewshed_arc_length(D_rotor, h_hub, h_terrain):
    arc_length = viewshed.calculate_viewshed_arc_length(
        D_rotor, h_hub, h_terrain=h_terrain
    )
    assert arc_length > 0
    # small angle approximation should be less than or equal to the full arc length
    arc_length_small = viewshed.calculate_viewshed_arc_length_smallangle(
        D_rotor, h_hub, h_terrain=h_terrain
    )
    assert arc_length_small <= arc_length


## COMPONENT TESTING


@pytest.fixture
def modeling_options():
    return {
        "windIO_plant": {
            "wind_farm": {
                "turbine": {
                    "rotor_diameter": 100,
                    "hub_height": 80,
                }
            }
        },
        "layout": {"N_turbines": 3},
    }


# make sure the viewshed of three overlapping turbines is less the sum of three
# turbines individual viewsheds but only slightly more than the viewshed of a
# single turbine
def test_viewshed_area_comp_overlap(modeling_options, subtests):

    # create the component to test
    comp = viewshed.ViewshedAreaComp(modeling_options=modeling_options)
    comp.setup()
    comp.setup_partials()

    # place turbines close so circles overlap
    x_turbines = np.array([0.0, 100.0, 200.0])
    y_turbines = np.array([0.0, 0.0, 0.0])
    inputs = {
        "x_turbines": x_turbines,
        "y_turbines": y_turbines,
    }
    outputs = {"area_viewshed": 0.0}

    # make sure the area of the viewshed computed return a plausible value
    comp.compute(inputs, outputs)

    # area should be less than sum of individual circles
    D_rotor = modeling_options["windIO_plant"]["wind_farm"]["turbine"]["rotor_diameter"]
    h_hub = modeling_options["windIO_plant"]["wind_farm"]["turbine"]["hub_height"]
    R_viewshed = viewshed.calculate_viewshed_arc_length(D_rotor, h_hub)
    expected_area_nonoverlapping = 3 * np.pi * R_viewshed**2 / 1e6
    expected_area_single_turbine = np.pi * R_viewshed**2 / 1e6
    expected_area_pyrite = 5211.7651786  # computed 28 Oct 2025

    with subtests.test("area less than non-overlapping sum"):
        assert outputs["area_viewshed"] < expected_area_nonoverlapping
    with subtests.test("area greater than single turbine viewshed"):
        assert outputs["area_viewshed"] > expected_area_single_turbine
    with subtests.test("area matches pyrite value"):
        assert np.isclose(outputs["area_viewshed"], expected_area_pyrite)


# make sure a single turbine viewshed is correct
def test_viewshed_area_comp_single_turbine(modeling_options):

    # create the component to test
    modeling_options["layout"]["N_turbines"] = 1  # modify the modeling options
    comp = viewshed.ViewshedAreaComp(modeling_options=modeling_options)
    comp.setup()
    comp.setup_partials()

    # place a single turbine at the origin
    x_turbines = np.array([0])
    y_turbines = np.array([0])
    inputs = {
        "x_turbines": x_turbines,
        "y_turbines": y_turbines,
    }
    outputs = {"area_viewshed": 0.0}

    # compute the expected area for a single turbine (area of a circle)
    D_rotor = modeling_options["windIO_plant"]["wind_farm"]["turbine"]["rotor_diameter"]
    h_hub = modeling_options["windIO_plant"]["wind_farm"]["turbine"]["hub_height"]
    R_viewshed = viewshed.calculate_viewshed_arc_length(D_rotor, h_hub)
    n_segs = 16
    expected_area = (
        4 * n_segs * (0.5 * R_viewshed**2 * np.sin(2 * np.pi / (4 * n_segs))) / 1.0e6
    )  # area of a equal-segmented approximation from shapely
    # expected_area = np.pi * R_viewshed**2 / 1e6  # area of a circle

    # compute with the component and verify
    comp.compute(inputs, outputs)
    assert np.isclose(outputs["area_viewshed"], expected_area, rtol=1e-3)


# make sure a viewshed area of nonoverlapping turbines returns a 3x one turbine
def test_viewshed_area_comp_three_nonoverlapping_turbines(modeling_options):

    # create the component to test
    comp = viewshed.ViewshedAreaComp(modeling_options=modeling_options)
    comp.setup()
    comp.setup_partials()

    # place turbines far apart so circles don't overlap
    x_turbines = np.array([50000.0, 0.0, -50000.0])
    y_turbines = np.array([50000.0, -50000.0, 50000.0])
    inputs = {
        "x_turbines": x_turbines,
        "y_turbines": y_turbines,
    }
    outputs = {"area_viewshed": 0.0}

    # compute the expected area for three single turbines (via area of a circle)
    D_rotor = modeling_options["windIO_plant"]["wind_farm"]["turbine"]["rotor_diameter"]
    h_hub = modeling_options["windIO_plant"]["wind_farm"]["turbine"]["hub_height"]
    R_viewshed = viewshed.calculate_viewshed_arc_length(D_rotor, h_hub)
    n_segs = 16
    expected_area = 3 * (
        4 * n_segs * (0.5 * R_viewshed**2 * np.sin(2 * np.pi / (4 * n_segs))) / 1.0e6
    )  # area of an equal-segmented approximation from shapely
    # expected_area = 3 * np.pi * R_viewshed**2 / 1e6

    # compute with the component and verify
    comp.compute(inputs, outputs)
    assert np.isclose(outputs["area_viewshed"], expected_area, rtol=1e-3)
