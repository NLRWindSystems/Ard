# FLOWERS

# Michael LoCascio

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import flowers.tools as tl

###########################################################################
# Wind rose methods
###########################################################################


def plot_wind_rose(
    wind_rose,
    ax=None,
    color_map="viridis_r",
    ws_right_edges=np.array([5, 10, 15, 20, 25]),
    wd_bins=np.arange(0, 360, 15.0),
    legend_kwargs={},
):
    """
    Plots a wind rose showing the frequency of occurence
    of the specified wind direction and wind speed bins. If no axis is
    provided, a new one is created. (Copied from FLORIS)

    Args:
        ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
            on which the wind rose is plotted. Defaults to None.
        color_map (str, optional): Colormap to use. Defaults to 'viridis_r'.
        ws_right_edges (np.array, optional): The upper bounds of the wind
            speed bins (m/s). The first bin begins at 0. Defaults to
            np.array([5, 10, 15, 20, 25]).
        wd_bins (np.array, optional): The wind direction bin centers used
            for plotting (deg). Defaults to np.arange(0, 360, 15.).
        legend_kwargs (dict, optional): Keyword arguments to be passed to
            ax.legend().

    Returns:
        :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
        the plotted wind rose.

    """
    # Resample data onto bins
    wr = wind_rose.copy(deep=True)
    df_plot = tl.resample_wind_direction(wr, wd=wd_bins)

    # Make labels for wind speed based on edges
    ws_step = ws_right_edges[1] - ws_right_edges[0]
    ws_labels = ["%d-%d m/s" % (w - ws_step, w) for w in ws_right_edges]

    # Grab the wd_step
    wd_step = wd_bins[1] - wd_bins[0]

    # Set up figure
    if ax is None:
        _, ax = plt.subplots(subplot_kw=dict(polar=True))

    # Get a color array
    color_array = cm.get_cmap(color_map, len(ws_right_edges))

    for wd_idx, wd in enumerate(wd_bins):
        rects = list()
        df_plot_sub = df_plot[df_plot.wd == wd]
        for ws_idx, ws in enumerate(ws_right_edges[::-1]):
            plot_val = df_plot_sub[
                df_plot_sub.ws <= ws
            ].freq_val.sum()  # Get the sum of frequency up to this wind speed
            rects.append(
                ax.bar(
                    np.radians(wd),
                    plot_val,
                    width=0.9 * np.radians(wd_step),
                    color=color_array(ws_idx),
                    edgecolor="k",
                    linewidth=0.5,
                    label=ws_labels[ws_idx],
                )
            )
        # break

    # Configure the plot
    ax.legend(
        reversed(rects),
        ws_labels,
        loc="lower left",
        bbox_to_anchor=(0.55 + np.cos(0.55) / 2, 0.4 + np.sin(0.55) / 2),
        **legend_kwargs,
    )
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_zero_location("N")
    ax.tick_params(axis="x", which="major", pad=-1)  # 1 for AEP paper
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_yticklabels([])
    ax.set_axisbelow(True)

    return ax


###########################################################################
# Layout methods
###########################################################################


def plot_layout(
    layout_x, layout_y, D=126.0, boundaries=None, norm=True, ax=None, color="tab:blue"
):
    """
    Plot a wind farm layout. The turbine markers are properly scaled
    relative to the domain.

    Args:
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        D (float): rotor diameter [m]
        norm (bool): dictates whether the plot should be scaled by
            rotor diameter. Defaults to True.
        ax (:py:class:`matplotlib.pyplot.axes`, optional): axis to
            plot layout

    Returns:
        ax (:py:class:`matplotlib.pyplot.axes`, optional): axis
            after plotting and formatting

    """

    if ax is None:
        _, ax = plt.subplots()

    if norm:
        xx = layout_x / D
        yy = layout_y / D
        r = 0.5
        xlab = "x/D"
        ylab = "y/D"

    else:
        xx = layout_x
        yy = layout_x
        r = D / 2
        xlab = "x [m]"
        ylab = "y [m]"

    if boundaries is not None:
        verts = np.array(boundaries) / D
        for i in range(len(verts)):
            if i == len(verts) - 1:
                ax.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "black")
            else:
                ax.plot(
                    [verts[i][0], verts[i + 1][0]],
                    [verts[i][1], verts[i + 1][1]],
                    "black",
                )

    ax.scatter(xx, yy, s=0.01)
    for x, y in zip(xx, yy):
        ax.add_patch(plt.Circle((x, y), r, color=color))
    ax.set(xlabel=xlab, ylabel=ylab, aspect="equal")
    ax.grid()

    return ax


def plot_optimal_layout(
    boundaries=[],
    x_final=[],
    y_final=[],
    x_init=[],
    y_init=[],
    D=126.0,
    color_initial="tab:blue",
    color_final="tab:orange",
    norm=True,
    ax=None,
):
    """
    Plot the initial and final solution of a layout optimization study.
    The turbine markers are properly scaled relative to the domain.

    Args:
        boundaries (list(tuple)): boundary vertices in the form
                [(x0,y0), (x1,y1), ... , (xN,yN)]
        x_final (numpy.array(float)): x-positions of each turbine
            in the optimal solution [m]
        y_final (numpy.array(float)): y-positions of each turbine
            in the optimal solution [m]
        x_init (numpy.array(float)): x-positions of each turbine
            in the initial solution [m]
        y_init (numpy.array(float)): y-positions of each turbine
            in the initial solution [m]
        D (float): rotor diameter [m]
        norm (bool): dictates whether the plot should be scaled by
            rotor diameter. Defaults to True.
        ax (:py:class:`matplotlib.pyplot.axes`, optional): axis to
            plot layout

    Returns:
        ax (:py:class:`matplotlib.pyplot.axes`, optional): axis
            after plotting and formatting

    """

    if ax is None:
        _, ax = plt.subplots()

    if norm:
        x1 = x_final / D
        y1 = y_final / D
        verts = np.array(boundaries) / D
        r = 0.5
        xlab = "x/D"
        ylab = "y/D"

    else:
        x1 = x_final
        y1 = y_final
        verts = boundaries
        r = D
        xlab = "x [m]"
        ylab = "y [m]"

    # Plot turbine locations
    ax.scatter(x1, y1, s=0.01, color=color_final)
    for x, y in zip(x1, y1):
        ax.add_patch(plt.Circle((x, y), r, color=color_final))
    ax.set(xlabel=xlab, ylabel=ylab, aspect="equal")
    ax.legend(["Final"], markerscale=50)
    ax.grid()

    if verts.ndim == 2:
        # Plot plant boundary for single boundary
        for i in range(len(verts)):
            if i == len(verts) - 1:
                ax.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "black")
            else:
                ax.plot(
                    [verts[i][0], verts[i + 1][0]],
                    [verts[i][1], verts[i + 1][1]],
                    "black",
                )

    else:
        # Plot plant boundary for multiple boundaries
        for ii in range(len(verts)):
            sub_verts = verts[ii]
            for i in range(len(sub_verts)):
                if i == len(sub_verts) - 1:
                    ax.plot(
                        [sub_verts[i][0], sub_verts[0][0]],
                        [sub_verts[i][1], sub_verts[0][1]],
                        "black",
                    )
                else:
                    ax.plot(
                        [sub_verts[i][0], sub_verts[i + 1][0]],
                        [sub_verts[i][1], sub_verts[i + 1][1]],
                        "black",
                    )

    return ax


###########################################################################
# Optimization performance methods
###########################################################################


def animate_layout_history(
    filename=None,
    layout_x=[],
    layout_y=[],
    boundaries=[],
    D=126.0,
    norm=True,
    show=True,
):
    """
    Animate the history of the wind farm layout. Plots the wind farm
    layout at each iteration and saves to the given file as an MP4.

    Args:
        filename (str, '.mp4'): name of animation file
        layout_x (tuple(numpy.array)): container of x-positions at each
                iteration.
        layout_y (tuple(numpy.array)): container of y-positions at each
                iteration.
        boundaries (list(tuple)): boundary vertices in the form
                [(x0,y0), (x1,y1), ... , (xN,yN)]
        D (float): rotor diameter [m]
        norm (bool): dictates whether the plot should be scaled by
            rotor diameter. Defaults to True.
        show (bool): dictates whether the animation is displayed before
            closing. Defaults to True.

    """

    if ax is None:
        _, ax = plt.subplots()

    if filename is None:
        raise ValueError("Must supply file name for layout history animation.")

    if norm:
        x = layout_x / D
        y = layout_y / D
        verts = boundaries / D
        xlab = "x/D"
        ylab = "y/D"

    else:
        x = layout_x
        y = layout_y
        verts = boundaries
        xlab = "x [m]"
        ylab = "y [m]"

    # Layout animation
    fig, ax = plt.subplots()
    ax.set(xlabel=xlab, ylabel=ylab, aspect="equal")
    ax.grid()

    for i in range(len(verts)):
        if i == len(verts) - 1:
            ax.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "black")
        else:
            ax.plot(
                [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "black"
            )

    (line,) = ax.plot([], [], "o")

    # Function to update turbine positions
    def animate(i):
        line.set_data(x[i], y[i])
        ax.set_title(str(i))
        return (line,)

    # Animation
    ani = animation.FuncAnimation(fig, animate, frames=len(x), repeat=False)
    ani.save(filename)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence_history(
    aep=[],
    optimality=[],
    feasibility=[],
    ax_aep=None,
    ax_opt=None,
    ax_feas=None,
):
    """
    Plots the convergence history of AEP (objective function),
    optimality, and feasibility. Axes should be supplied for any
    metrics that are to be plotted.

    Args:
        aep (list(float)): AEP at each major iteration
        optimality (list(float)): SNOPT optimality at each major iteration
        feasibility (list(float)): SNOPT feasibility at each major iteration
        ax_aep (:py:class:`matplotlib.pyplot.axes`, optional): axis to
            plot AEP history
        ax_opt (:py:class:`matplotlib.pyplot.axes`, optional): axis to
            plot optimality history
        ax_feas (:py:class:`matplotlib.pyplot.axes`, optional): axis to
            plot feasibility history

    """

    # Objective plot
    if len(aep) > 0:
        if ax_aep is None:
            _, ax_aep = plt.subplots()

        ax_aep.plot([elem / 1e9 for elem in aep])
        ax_aep.set(xlabel="Iteration", ylabel="AEP [GWh]")
        ax_aep.grid(True)

    # Optimality plot
    if len(optimality) > 0:
        if ax_opt is None:
            _, ax_opt = plt.subplots()

        ax_opt.semilogy(optimality)
        ax_opt.set(xlabel="Iteration", ylabel="Optimality [-]")
        ax_opt.grid(True)

    # Feasibility plot
    if len(feasibility) > 0:
        if ax_feas is None:
            _, ax_feas = plt.subplots()

        ax_feas.semilogy(feasibility)
        ax_feas.set(xlabel="Iteration", ylabel="Feasibility [-]")
        ax_feas.grid(True)


## Legacy functions


def plot_constraints(ax_boundary, ax_spacing, boundary_constraint, spacing_constraint):
    """
    Plots the convergence history of the objective function and the wind farm
        layout (optional)

    Args:
        ax: matplotlib axis handle to plot AEP history.
        obj (list(float)): A list of AEP at each major iteration.
        layout (tuple(float)): A list of wind farm (x,y) layout at each major iteration.
        boundaries (list(float)): A list of the boundary vertices in the form
            [(x0,y0), (x1,y1), ... , (xN,yN)].
        D (float): rotor diameter
        filename (str): name of .mp4 animation of layout progression.
    """

    # Boundary constraint plot
    for n in range(len(boundary_constraint)):
        ax_boundary.plot(boundary_constraint[n], alpha=0.3)
    ax_boundary.set(xlabel="Iteration", ylabel="Boundary Constraint")
    ax_boundary.grid()

    # Spacing constraint plot
    ax_spacing.plot(spacing_constraint)
    ax_spacing.set(xlabel="Iteration", ylabel="Spacing Constraint")
    ax_spacing.grid()


def plot_flow_field(fi, ax, bounds, pts=200, cmin=2, cmax=10):
    """
    Plots a filled contour map of the annually-averaged flow field.

    Args:
        fi: FLOWERS interface with valid Fourier coefficients
        ax: matplotlib axis handle to plot colormesh
        bounds (np.array): domain limits in the form
            ([x_min, x_max], [y_min, y_max])
        pts: grid resolution (uniform in x and y)
        cmin: minimum wind speed for colorbar [m/s]
        cmax: maximum wind speed for colorbar [m/s]

    """

    # Enforce that fourier_coefficients() have been computed
    if not hasattr(fi, "fs"):
        print("Error, must compute Fourier coefficients before calculating wake")
        return None

    # Define 2D grid based on defined domain limits
    xx = np.linspace(bounds[0, 0], bounds[0, 1], pts)
    yy = np.linspace(bounds[1, 0], bounds[1, 1], pts)
    XX, YY = np.meshgrid(xx, yy)

    # Freestream velocity component
    u0 = fi.fs.a_free[0] * np.pi

    # Superimpose wake velocity component from each turbine
    for i in range(len(fi.layout_x)):
        if i == 0:
            du = fi.calculate_wake(XX - fi.layout_x[i], YY - fi.layout_y[i])
        else:
            du += fi.calculate_wake(XX - fi.layout_x[i], YY - fi.layout_y[i])

    # Compute average wake velocity
    u = u0 - du

    # Mask points within rotor swept area
    zz = np.logical_or(
        np.isnan(u),
        np.sqrt((XX - fi.layout_x[0]) ** 2 + (YY - fi.layout_y[0]) ** 2) < fi.D / 2,
    )
    for j in range(len(fi.layout_x) - 1):
        zz = np.logical_or(
            zz,
            np.sqrt((XX - fi.layout_x[j + 1]) ** 2 + (YY - fi.layout_y[j + 1]) ** 2)
            < fi.D / 2,
        )
    u_masked = np.ma.masked_where(zz, u)

    # Plot wake velocity colormesh and rotor swept areas
    im = ax.pcolormesh(XX, YY, u_masked, vmin=cmin, vmax=cmax, cmap="coolwarm")
    ax.plot(fi.layout_x, fi.layout_y, "ow")

    return im


def plot_floris_field(fli, ax, wind_rose, bounds, pts=200, cmin=2, cmax=10):
    """
    Plots a filled contour map of the annually-averaged flow field using FLORIS.

    Args:
        fli: FLORIS interface
        fi: FLOWERS interface with valid Fourier coefficients
        ax: matplotlib axis handle to plot colormesh
        bounds (np.array): domain limits in the form
            ([x_min, x_max], [y_min, y_max])
        pts: grid resolution (uniform in x and y)
        cmin: minimum wind speed for colorbar [m/s]
        cmax: maximum wind speed for colorbar [m/s]
    """

    # Resample wind rose by average wind speed to speed up plotting
    wr = wind_rose.copy(deep=True)
    wr = tl.resample_wind_direction(wr, wd=np.arange(0, 360, 5.0))
    wr = tl.resample_average_ws_by_wd(wr)

    # Numerically integrate over wind rose
    for i in range(len(wr.wd)):  # range(len(wr.wd))

        # Redefine flow field for given wind speed and direction
        fli.reinitialize(wind_directions=[wr.wd[i]], wind_speeds=[wr.ws[i]])
        if i == 0:
            hor_plane = fli.calculate_horizontal_plane(
                height=90.0,
                x_resolution=pts,
                y_resolution=pts,
                x_bounds=[bounds[0, 0], bounds[0, 1]],
                y_bounds=[bounds[1, 0], bounds[1, 1]],
            )
            hor_plane.df.u = wr.freq_val[i] * hor_plane.df.u
        else:
            hor_plane1 = fli.calculate_horizontal_plane(
                height=90.0,
                x_resolution=pts,
                y_resolution=pts,
                x_bounds=[bounds[0, 0], bounds[0, 1]],
                y_bounds=[bounds[1, 0], bounds[1, 1]],
            )
            hor_plane.df.u += wr.freq_val[i] * hor_plane1.df.u

    # Reshape mesh grid for plotting
    x1_mesh = hor_plane.df.x1.values.reshape(
        hor_plane.resolution[1], hor_plane.resolution[0]
    )
    x2_mesh = hor_plane.df.x2.values.reshape(
        hor_plane.resolution[1], hor_plane.resolution[0]
    )
    u_mesh = hor_plane.df.u.values.reshape(
        hor_plane.resolution[1], hor_plane.resolution[0]
    )

    # Mask rotor swept areas (add rotor diameter variable)
    zz = np.logical_or(
        np.isnan(u_mesh),
        np.sqrt(
            (x1_mesh - fli.floris.farm.layout_x[0]) ** 2
            + (x2_mesh - fli.floris.farm.layout_y[0]) ** 2
        )
        < 126 / 2,
    )
    for j in range(len(fli.floris.farm.layout_x) - 1):
        zz = np.logical_or(
            zz,
            np.sqrt(
                (x1_mesh - fli.floris.farm.layout_x[j + 1]) ** 2
                + (x2_mesh - fli.floris.farm.layout_y[j + 1]) ** 2
            )
            < 126 / 2,
        )
    u = np.ma.masked_where(zz, u_mesh)

    # Plot wake velocity colormesh and rotor swept areas
    im = ax.pcolormesh(x1_mesh, x2_mesh, u, cmap="coolwarm")  # vmin=cmin, vmax=cmax,
    for j in range(len(fli.floris.farm.layout_x)):
        ax.scatter(fli.floris.farm.layout_x[j], fli.floris.farm.layout_y[j], c="white")

    return im
