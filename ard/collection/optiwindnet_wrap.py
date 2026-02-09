from warnings import warn

import networkx as nx
import numpy as np

from optiwindnet.mesh import make_planar_embedding
from optiwindnet.interarraylib import L_from_site
from optiwindnet.heuristics import EW_presolver
from optiwindnet.MILP import OWNWarmupFailed, solver_factory, ModelOptions

from . import templates


def _own_L_from_inputs(inputs: dict, discrete_inputs: dict) -> nx.Graph:
    # get the metadata and data for the OWN warm-starter from the inputs
    T = len(inputs["x_turbines"])
    R = len(inputs["x_substations"])
    name_case = "farm"
    if discrete_inputs["x_border"] is not None:
        B = len(discrete_inputs["x_border"])
    else:
        B = 0
    VertexC = np.empty((R + T + B, 2), dtype=float)
    VertexC[:T, 0] = inputs["x_turbines"]
    VertexC[:T, 1] = inputs["y_turbines"]
    VertexC[-R:, 0] = inputs["x_substations"]
    VertexC[-R:, 1] = inputs["y_substations"]

    # add perturbation to duplicate turbine/substation positions
    VertexCTR = np.vstack([VertexC[:T, :], VertexC[-R:, :]])
    perturbation_eps = 1.0e-6  # base magnitude of perturbation in m
    perturbation_normal = np.array([-1.0, 1.0])  # set a fixed axis to perturb on
    perturbation_normal = perturbation_normal / np.sqrt(
        np.sum(perturbation_normal**2)
    )  # normalize the perturbation
    # go through the turbine/substation vertices and count how many times a
    # given vertex has appeared before
    repeat_accumulate = np.array(
        [
            int(np.sum(np.all(VertexCTR[:ivv, :] == vv, axis=1)))
            for ivv, vv in enumerate(VertexCTR)
        ]
    )
    if np.any(repeat_accumulate > 0):  # only if there are any repeats
        prefix = "\n"
        warn_string = prefix.join(
            f"Detected {np.sum(repeat_accumulate > 0)} coincident "
            f"turbines and/or substations in optiwindnet setup."
        )  # start a warning string for the UserWarning
        # TODO: make Ard warnings?

        # create perturbation adjustements s.t. vertices w/ multiplicity > 2
        # are adjusted to be fully unique!
        adjustments = perturbation_eps * np.outer(
            repeat_accumulate, perturbation_normal
        )
        # for each adjustments add to the warning string
        for idx, dxy in enumerate(adjustments[:T, :]):
            if np.sum(dxy != 0) == 0:
                continue
            prefix = "\n\t"
            warn_string += prefix.join(
                f"adjusting turbine #{idx} from {VertexCTR[idx, :]} to  {VertexCTR[idx, :] + dxy}"
            )
        for idx, dxy in enumerate((adjustments[-R:, :])[::-1, :]):
            if np.sum(dxy != 0) == 0:
                continue
            prefix = "\n\t"
            warn_string += prefix.join(
                f"adjusting substation #{idx} from {VertexCTR[-(idx+1), :]} to {VertexCTR[-(idx+1), :] + dxy}"
            )
        # output the final warning
        warn(warn_string)

        # store the adjustments
        VertexCTR += adjustments

    # apply the adjustments
    VertexC[:T, :] = VertexCTR[:T, :]
    VertexC[-R:, :] = VertexCTR[-R:, :]

    # put together the inputs for optiwindnet
    site = dict(
        T=T,
        R=R,
        name=name_case,
        handle=name_case,
        VertexC=VertexC,
    )

    # handle the boundary if it exists
    if B > 0:
        VertexC[T:-R, 0] = discrete_inputs["x_border"]
        VertexC[T:-R, 1] = discrete_inputs["y_border"]
        site["B"] = B
        site["border"] = np.arange(T, T + B)
    return L_from_site(**site)


class OptiwindnetCollection(templates.CollectionTemplate):
    """
    Component class for modeling optiwindnet-optimized energy collection systems.

    A component class to make a heuristic-based optimized energy collection and
    management system using optiwindnet! Inherits the interface from
    `templates.CollectionTemplate`.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines`
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines`
    x_substations : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the substations,
        with length `N_substations`
    y_substations : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the substations,
        with length `N_substations`

    Outputs
    -------
    total_length_cables : float
        the total length of cables used in the collection system network

    Discrete Outputs
    -------
    length_cables : np.ndarray
        a 1D numpy array that holds the lengths of each of the cables necessary
        to collect energy generated, with length `N_turbines`
    load_cables : np.ndarray
        a 1D numpy array that holds the turbine count upstream of the cable segment
        (i.e. number of turbines whose power is collected through the cable), with
        length `N_turbines`
    max_load_cables : int
        the maximum cable capacity required by the collection system
    terse_links : np.ndarray
        a 1D numpy int array encoding the electrical connections of the collection
        system (tree topology), with length `N_turbines`
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()
        self.S_previous: nx.Graph | None = None

    def setup(self):
        """Setup of OM component."""
        super().setup()

    def setup_partials(self):
        """Setup of OM component gradients."""

        self.declare_partials(
            ["total_length_cables"],
            ["x_turbines", "y_turbines", "x_substations", "y_substations"],
            method="exact",
        )

    def compute(
        self,
        inputs,
        outputs,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        Computation for the OptiWindNet collection system design
        """

        max_turbines_per_string = self.modeling_options["collection"][
            "max_turbines_per_string"
        ]
        solver_name = self.modeling_options["collection"]["solver_name"]

        # get a graph representing the updated location
        L = _own_L_from_inputs(inputs, discrete_inputs)
        T = L.graph["T"]

        # create planar embedding and set of available links
        P, A = make_planar_embedding(L)

        solver = solver_factory(solver_name)

        model_options = self.modeling_options["collection"]["model_options"]
        # start from previous solution if available, else from heuristic if it fits
        if self.S_previous is not None:
            S_warm = self.S_previous
        elif (
            model_options.get("topology") == "branched"
            and model_options.get("feeder_limit") == "unlimited"
            and model_options.get("feeder_route") == "segmented"
        ):
            S_warm = EW_presolver(A, capacity=max_turbines_per_string)
        else:
            S_warm = None

        try:
            solver.set_problem(
                P,
                A,
                max_turbines_per_string,
                ModelOptions(**model_options),
                warmstart=S_warm,
            )
        except OWNWarmupFailed:
            # the previous solution is no longer feasible
            solver.set_problem(
                P,
                A,
                max_turbines_per_string,
                ModelOptions(**model_options),
            )

        # do the branch-and-bound MILP search
        info = solver.solve(**self.modeling_options["collection"]["solver_options"])
        S, G = solver.get_solution()
        self.S_previous = S

        # extract the outputs
        terse_links = np.zeros((T,), dtype=np.int_)
        length_cables = np.zeros((T,))
        load_cables = np.zeros((T,))

        d2roots = A.graph["d2roots"]
        # convert the graph to array representing the tree (edges i->terse[i])
        for u, v, edgeD in S.edges(data=True):
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if edgeD["reverse"] else (v, u)
            terse_links[i] = target
            load = edgeD["load"]
            load_cables[i] = load
            if u < 0:
                # u is a substation
                if v in G[u]:
                    # feeder <u, v> has a straight route
                    length_cables[i] = d2roots[v, u]
                else:
                    # feeder <u, v> is segmented (detoured route)
                    v_neighbors = G[v]
                    for cur_hop in v_neighbors:
                        if cur_hop >= T and v_neighbors[cur_hop]["load"] == load:
                            break
                    length_cables[i] = v_neighbors[cur_hop]["length"]
                    prev_hop = v
                    while cur_hop >= T:
                        s, t = G[cur_hop]
                        cur_hop, prev_hop = (s if t == prev_hop else t), cur_hop
                        length_cables[i] += G[cur_hop][prev_hop]["length"]
            else:
                # link (u, v) is not a feeder, so A has length data
                length_cables[i] = A[u][v]["length"]

        # pack and ship
        self.graph = G
        discrete_outputs["graph"] = G  # TODO: remove for terse links, below!
        discrete_outputs["terse_links"] = terse_links
        discrete_outputs["length_cables"] = length_cables
        discrete_outputs["load_cables"] = load_cables
        discrete_outputs["max_load_cables"] = S.graph["max_load"]
        # TODO: remove this assert after enough testing
        assert (
            abs(length_cables.sum() - G.size(weight="length")) < 1e-7
        ), f"difference: {length_cables.sum() - G.size(weight='length')}"
        outputs["total_length_cables"] = length_cables.sum()

    def compute_partials(self, inputs, J, discrete_inputs=None):

        # re-load the key variables back as locals
        G = self.graph
        T = G.graph["T"]
        R = G.graph["R"]
        VertexC = G.graph["VertexC"]
        gradients = np.zeros_like(VertexC)

        fnT = G.graph.get("fnT")
        if fnT is not None:
            _u, _v = fnT[np.array(G.edges)].T
        else:
            _u, _v = np.array(G.edges).T
        vec = VertexC[_u] - VertexC[_v]
        norm = np.hypot(*vec.T)
        # suppress the contributions of zero-length edges
        norm[np.isclose(norm, 0.0)] = 1.0
        vec /= norm[:, None]

        np.add.at(gradients, _u, vec)
        np.subtract.at(gradients, _v, vec)

        # wind turbines
        J["total_length_cables", "x_turbines"] = gradients[:T, 0]
        J["total_length_cables", "y_turbines"] = gradients[:T, 1]

        # substations
        J["total_length_cables", "x_substations"] = gradients[-R:, 0]
        J["total_length_cables", "y_substations"] = gradients[-R:, 1]

        return J
