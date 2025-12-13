import os

import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt

import openmdao

import optiwindnet.plotting


# get plot limits based on the farm boundaries
def get_limits(
    windIOdict: dict,
    lim_buffer: float = 0.05,
):
    """
    generate plot limits based on the expected values found in the windIO file

    Parameters
    ----------
    windIOplant : dict
        a full, presumed validated, windIO plant specification file
    lim_buffer : float, optional
        a percent buffer for plot edges, by default 0.05 (5%)

    Returns
    -------
    x_lim : np.ndarray
        the two-valued limits for the x-axis based on the windIO
    y_lim : np.ndarray
        the two-valued limits for the y-axis based on the windIO
    """

    x_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
    ]
    y_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
    ]
    return x_lim, y_lim


def plot_layout(
    ard_prob: openmdao.api.Problem,
    input_dict: dict,
    ax: matplotlib.axes.Axes = None,
    show_image: bool = False,
    save_path: os.PathLike = None,
    save_kwargs: dict = {},
    include_cable_routing: bool = False,
    include_mooring_system: bool = False,
):
    """
    plot the layout of a farm

    Parameters
    ----------
    ard_prob : openmdao.api.Problem
        the active Ard/OpenMDAO problem
    input_dict : dict
        the active Ard input dictionary
    ax : matplotlib.axes.Axes, optional
        an already-active pyplot Axes, by default None
    show_image : bool, optional
        to show the image, rather than just saving, by default False
    save_path : os.PathLike, optional
        location where the image be saved, by default None
    save_kwargs : dict, optional
        optional keyword arguments for plt.savefig, by default {}
    include_cable_routing : bool, optional
        should the collection system routing be plotted also, by default False

    Returns
    -------
    matplotlib.axes.Axes
        the matplotlib Axes that have been generated (or modified)
    """

    # get the turbine locations to plot
    x_turbines = ard_prob.get_val("x_turbines", units="m")
    y_turbines = ard_prob.get_val("y_turbines", units="m")

    # make axis object
    if ax is None:
        fig, ax = plt.subplots()

    # plot wind plant boundaries
    windIO_dict = input_dict["modeling_options"]["windIO_plant"]

    ax.fill(
        windIO_dict["site"]["boundaries"]["polygons"][0]["x"],
        windIO_dict["site"]["boundaries"]["polygons"][0]["y"],
        linestyle="--",
        alpha=0.5,
        fill=False,
        c="k",
        # linecolor="k",
    )

    # loop over the exclusion types
    for excl_type_key, excl_type_value in (
        windIO_dict["site"].get("exclusions", {}).items()
    ):
        # handle un-implemented exclusion types
        if excl_type_key not in ["polygons"]:
            raise NotImplementedError(
                f"`{excl_type_key}` exclusions found but not yet handled in Ard plotting capabilities."
            )

        if excl_type_key == "polygons":
            for polygon_exclusion in excl_type_value:
                ax.fill(
                    polygon_exclusion["x"],
                    polygon_exclusion["y"],
                    linestyle="--",
                    alpha=0.25,
                    fill=True,
                    c="r",
                    # linecolor="k",
                )
        else:
            raise ValueError("this line should be inaccessible.")

    # plot turbines
    ax.plot(x_turbines, y_turbines, "ok")

    # adjust plot limits
    x_lim, y_lim = get_limits(windIO_dict)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if include_cable_routing:
        optiwindnet.plotting.gplot(
            ard_prob.model.collection.graph,
            ax=ax,
            dark=False,
            legend=False,
            hide_ST=True,
            infobox=False,
            landscape=False,
        )

    if include_mooring_system:
        # get the coordinates of the anchors
        x_anchors = ard_prob.get_val("x_anchors", units="m")
        y_anchors = ard_prob.get_val("y_anchors", units="m")

        # loop over the anchors and plot from their originating turbine to each
        for idx_turbine in range(
            input_dict["modeling_options"]["layout"]["N_turbines"]
        ):
            for idx_anchor in range(
                input_dict["modeling_options"]["platform"]["N_anchors"]
            ):
                ax.plot(
                    [x_turbines[idx_turbine], x_anchors[idx_turbine, idx_anchor]],
                    [y_turbines[idx_turbine], y_anchors[idx_turbine, idx_anchor]],
                    "-r",
                    alpha=0.25,
                )
            # plot the anchors as red circles
            ax.plot(
                x_anchors[idx_turbine, :],
                y_anchors[idx_turbine, :],
                "or",
                alpha=0.25,
            )

    ax.axis("equal")

    # show, save, or return
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    if show_image:
        plt.show()

    return ax
