from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd

import openmdao.api as om

import wisdem.orbit.orbit_api as orbit_wisdem

from ORBIT.core.library import default_library
from ORBIT.core.library import initialize_library

from ard.cost.wisdem_wrap import ORBIT_setup_latents


def generate_orbit_location_from_graph(
    graph,  # TODO: replace with a terse_links representation
    X_turbines,
    Y_turbines,
    X_substations,
    Y_substations,
    allow_branching_approximation=False,
):
    """
    go from a optiwindnet graph to an ORBIT input CSV

    convert a optiwindnet graph representation of a collection system and get to
    a best-possible approximation of the same collection system for
    compatibility with ORBIT. ORBIT doesn't allow branching and optiwindnet does
    by default, so we allow some cable duplication if necessary to get a
    conservative approximation of the BOS costs if the graph isn't compatible
    with ORBIT

    Parameters
    ----------
    graph : networkx.Graph
        the graph representation of the collection system design
    X_turbines : np.array
        the cartesian X locations, in kilometers, of the turbines
    Y_turbines : np.array
        the cartesian Y locations, in kilometers, of the turbines
    X_substations : np.array
        the cartesian X locations, in kilometers, of the substations
    Y_substations : np.array
        the cartesian Y locations, in kilometers, of the substations

    Returns
    -------
    pandas.DataFrame
        a dataframe formatted for ORBIT to specify a farm layout

    Raises
    ------
    RecursionError
        if the recursive setup seems to be stuck in a loop
    """

    # get all edges, sorted by the first node then the second node
    edges_to_process = [edge for edge in graph.edges]
    edges_to_process.sort(key=lambda x: (x[0], x[1]))
    # get the edges with a negative index node (a substation)
    edges_inclsub = [edge for edge in edges_to_process if edge[0] < 0 or edge[1] < 0]
    edges_inclsub.sort(key=lambda x: (x[0], x[1]))

    # check to see if any nodes appear more than twice
    # (i.e. once destination and possibly one source)
    node_countmap = dict.fromkeys(
        list(set(node for edge in edges_to_process for node in edge)), 0
    )
    for edge in edges_to_process:
        node_countmap[edge[0]] += 1
        node_countmap[edge[1]] += 1
    # if this has branching, handle it
    if np.any(
        (
            np.array(list(node_countmap.values())) > 2
        )  # multiple turbine appearances indicates a branch
        & (
            np.array(list(node_countmap.keys())) >= 0
        )  # but substations do appear so "mask" them
    ):
        if allow_branching_approximation:
            warnings.warn(
                "The provided collection system design graph includes branching, "
                "which ORBIT does not support. Proceeding with an approximate "
                "radial collection system for cost modeling."
            )
        else:
            raise ValueError(
                "The graph has branching. ORBIT does not support this. "
                "By modifying the approximate_branches option to True in the "
                "ORBITDetail component, you can allow ORBIT to approximate the "
                "BOS costs by a close radial-layout collection system "
                "approximation."
            )

    # data for ORBIT
    data_orbit = {
        "id": [],
        "substation_id": [],
        "name": [],
        "longitude": [],
        "latitude": [],
        "string": [],
        "order": [],
        "cable_length": [],
        "bury_speed": [],
    }

    idx_string = 0
    order = 0

    for edge in edges_inclsub:  # every edge w/ a substation starts a string

        def handle_edge(
            edge, turbine_origination, idx_string, order, recursion_level=0
        ):
            # recursively handle the edges

            if recursion_level > 10:  # for safe recursion
                raise RecursionError("Recursion limit reached")

            # get the target turbine index
            turbine_tgt_index = edge[0] if edge[0] != turbine_origination else edge[1]
            # get the turbine name
            turbine_name = turbine_id = f"t{turbine_tgt_index:03d}"

            # add the turbine to the dataset
            data_orbit["id"].append(turbine_id)
            data_orbit["substation_id"].append(substation_id)
            data_orbit["name"].append(turbine_name)
            data_orbit["longitude"].append(X_turbines[turbine_tgt_index])
            data_orbit["latitude"].append(Y_turbines[turbine_tgt_index])
            data_orbit["string"].append(int(idx_string))
            data_orbit["order"].append(int(order))
            data_orbit["cable_length"].append(0)  # ORBIT computes automatically
            data_orbit["bury_speed"].append(0)  # ORBIT computes automatically

            # pop this edge out of the edges list
            edges_to_process.remove(edge)

            # get the set of remaining edges that include the terminal turbine
            edges_turbine = [e for e in edges_to_process if (turbine_tgt_index in e)]

            order += 1

            for new_string, edge_next in enumerate(edges_turbine):
                if new_string:
                    idx_string += 1
                    order = 0
                idx_string, order = handle_edge(
                    edge_next,
                    turbine_tgt_index,
                    idx_string,
                    order,
                    recursion_level=recursion_level + 1,
                )

            return idx_string, order

        # get the substation id as a one-liner
        substation_index = len(X_substations) + (edge[0] if edge[0] < 0 else edge[1])
        # get the substation name
        substation_name = substation_id = f"oss{substation_index:01d}"

        # add the substation to the dataset
        if not substation_id in data_orbit["id"]:
            data_orbit["id"].append(substation_id)
            data_orbit["substation_id"].append(substation_id)
            data_orbit["name"].append(substation_name)
            data_orbit["longitude"].append(X_substations[substation_index] / 1.0e3)
            data_orbit["latitude"].append(Y_substations[substation_index] / 1.0e3)
            data_orbit["string"].append(None)
            data_orbit["order"].append(None)
            data_orbit["cable_length"].append(None)
            data_orbit["bury_speed"].append(None)

        # handle the edge that we get
        idx_string, order = handle_edge(
            edge, substation_index - len(X_substations), idx_string, order
        )

        order = 0
        idx_string += 1

    df_orbit = pd.DataFrame(data_orbit).fillna("")
    df_orbit.string = [int(v) if v != "" else "" for v in df_orbit.string]
    df_orbit.order = [int(v) if v != "" else "" for v in df_orbit.order]

    return df_orbit


class ORBITDetail(orbit_wisdem.Orbit):
    """
    Wrapper for WISDEM's ORBIT offshore BOS calculators.

    A thicker wrapper of `wisdem.orbit_api` that 1) replaces capabilities that
    assume a grid farm layout that is default in WISDEM's ORBIT with a custom
    array layout, and 2) traps warning messages that are recognized not to be
    issues.

    See: https://github.com/NLRWindSystems/ORBIT
    """

    def initialize(self):
        """Initialize for API connections."""
        super().initialize()

        self.options.declare("case_title", default="working")
        self.options.declare("modeling_options")
        self.options.declare("approximate_branches", default=False)

    def setup(self):
        """Define all input variables from all models."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]
        self.N_substations = self.modeling_options["layout"]["N_substations"]

        self.set_input_defaults("wtiv", "example_wtiv")
        self.set_input_defaults("feeder", "example_feeder")
        # self.set_input_defaults("num_feeders", 1)
        # self.set_input_defaults("num_towing", 1)
        # self.set_input_defaults("num_station_keeping", 3)
        # self.set_input_defaults(
        #    "oss_install_vessel", "example_heavy_lift_vessel",
        # )
        self.set_input_defaults("site_distance", 40.0, units="km")
        self.set_input_defaults("site_distance_to_landfall", 40.0, units="km")
        self.set_input_defaults("interconnection_distance", 40.0, units="km")
        self.set_input_defaults("plant_turbine_spacing", 7)
        self.set_input_defaults("plant_row_spacing", 7)
        self.set_input_defaults("plant_substation_distance", 1, units="km")
        # self.set_input_defaults("num_port_cranes", 1)
        # self.set_input_defaults("num_assembly_lines", 1)
        self.set_input_defaults("takt_time", 170.0, units="h")
        self.set_input_defaults("port_cost_per_month", 2e6, units="USD/mo")
        self.set_input_defaults("construction_insurance", 44.0, units="USD/kW")
        self.set_input_defaults("construction_financing", 183.0, units="USD/kW")
        self.set_input_defaults("contingency", 316.0, units="USD/kW")
        self.set_input_defaults("commissioning_cost_kW", 44.0, units="USD/kW")
        self.set_input_defaults("decommissioning_cost_kW", 58.0, units="USD/kW")
        self.set_input_defaults("site_auction_price", 100e6, units="USD")
        self.set_input_defaults("site_assessment_cost", 50e6, units="USD")
        self.set_input_defaults("construction_plan_cost", 1e6, units="USD")
        self.set_input_defaults("installation_plan_cost", 2.5e5, units="USD")
        self.set_input_defaults("boem_review_cost", 0.0, units="USD")

        self.add_subsystem(
            "orbit",
            ORBITWisdemDetail(
                modeling_options=self.modeling_options,
                case_title=self.options["case_title"],
                approximate_branches=self.options["approximate_branches"],
                floating=self.modeling_options["floating"],
                jacket=self.modeling_options.get("jacket"),
                jacket_legs=self.modeling_options.get("jacket_legs"),
            ),
            promotes=["*"],
        )


class ORBITWisdemDetail(orbit_wisdem.OrbitWisdem):
    """ORBIT-WISDEM Fixed Substructure API, modified for detailed layouts"""

    _path_library = None

    def initialize(self):
        super().initialize()

        self.options.declare("case_title", default="working")
        self.options.declare("modeling_options")
        self.options.declare("approximate_branches", default=False)

    def setup(self):
        """Define all the inputs."""

        # call the superclass method
        super().setup()

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]
        self.N_substations = self.modeling_options["layout"]["N_substations"]

        # bring in collection system design
        self.add_discrete_input("graph", None)

        # add the detailed turbine and substation locations
        self.add_input("x_turbines", np.zeros((self.N_turbines,)), units="km")
        self.add_input("y_turbines", np.zeros((self.N_turbines,)), units="km")
        self.add_input("x_substations", np.zeros((self.N_substations,)), units="km")
        self.add_input("y_substations", np.zeros((self.N_substations,)), units="km")

        # copy the default ORBIT library to a local directory under case_files
        path_library_default = Path(default_library).absolute()
        self._path_library = (
            Path("case_files") / self.options["case_title"] / "ORBIT_library"
        ).absolute()
        if path_library_default.exists():
            shutil.copytree(
                path_library_default, self._path_library, dirs_exist_ok=True
            )
        else:
            raise FileNotFoundError(
                f"Can not find default ORBIT library at {path_library_default}."
            )

    def compile_orbit_config_file(
        self,
        inputs,
        outputs,
        discrete_inputs,
        discrete_outputs,
    ):

        config = super().compile_orbit_config_file(
            inputs,
            outputs,
            discrete_inputs,
            discrete_outputs,
        )  # run the superclass

        # remove the grid plant option, and replace with a custom plant
        config["plant"] = {
            "layout": "custom",
            "num_turbines": int(discrete_inputs["number_of_turbines"]),
            "turbine_spacing": inputs["plant_turbine_spacing"],
            "row_spacing": inputs["plant_row_spacing"],
        }

        # switch to the custom array system design
        if not ("ArraySystemDesign" in config["design_phases"]):
            raise KeyError(
                "I assumed that 'ArraySystemDesign' would be in the config. Something changed."
            )
        config["design_phases"][
            config["design_phases"].index("ArraySystemDesign")
        ] = "CustomArraySystemDesign"

        # add a turbine location csv on the config
        basename_farm_location = "wisdem_detailed_array"
        config["array_system_design"]["distance"] = True  # don't use WGS84 lat/long
        config["array_system_design"]["location_data"] = basename_farm_location
        config["array_system_design"]["cables"] = [
            f"XLPE_185mm_66kV{'_dynamic' if self.options['floating'] else ''}",
            f"XLPE_500mm_132kV{'_dynamic' if self.options['floating'] else ''}",
            f"XLPE_630mm_66kV{'_dynamic' if self.options['floating'] else ''}",
            f"XLPE_1000mm_220kV{'_dynamic' if self.options['floating'] else ''}",
        ]  # we require bigger cables than the standard WISDEM wrap

        # create the csv file that holds the farm layout
        path_farm_location = (
            self._path_library / "cables" / (basename_farm_location + ".csv")
        )

        # generate the csv data needed to locate the farm elements
        generate_orbit_location_from_graph(
            discrete_inputs["graph"],
            inputs["x_turbines"],
            inputs["y_turbines"],
            inputs["x_substations"],
            inputs["y_substations"],
            allow_branching_approximation=self.options["approximate_branches"],
        ).to_csv(path_farm_location, index=False)

        self._orbit_config = config  # reinstall- probably not needed due to reference
        return config  # and return

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Creates and runs the project, then gathers the results."""

        # setup the custom-location library
        if self._path_library:
            initialize_library(self._path_library)

        # send it back to the superclass compute
        super().compute(
            inputs,
            outputs,
            discrete_inputs,
            discrete_outputs,
        )


class ORBITDetailedGroup(om.Group):
    """wrapper for ORBIT-WISDEM Fixed Substructure API, allowing manual IVC incorporation"""

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare("case_title", default="working")
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )
        self.options.declare("approximate_branches", default=False)

    def setup(self):

        # add IVCs for landbosse
        variable_mapping = ORBIT_setup_latents(
            modeling_options=self.options["modeling_options"]
        )

        # create source independent variable components for LandBOSSE inputs
        for key, meta in variable_mapping.items():
            if key in ["number_of_turbines", "number_of_blades", "num_mooring_lines"]:
                comp = om.IndepVarComp()
                comp.add_discrete_output(name=key, val=meta["val"])
                self.add_subsystem(f"IVC_orbit_{key}", comp, promotes=["*"])
            else:
                self.add_subsystem(
                    f"IVC_orbit_{key}",
                    om.IndepVarComp(key, val=meta["val"], units=meta["units"]),
                    promotes=["*"],
                )

        # add orbit
        self.add_subsystem(
            "orbit",
            ORBITDetail(
                case_title=self.options["case_title"],
                modeling_options=self.options["modeling_options"],
                approximate_branches=self.options["approximate_branches"],
            ),
            promotes=[
                "total_capex",
                "total_capex_kW",
                "bos_capex",
                "installation_capex",
                "graph",
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
                # "plant_turbine_spacing",
                # "plant_row_spacing",
            ],
        )

        # connect
        for key in variable_mapping.keys():
            self.connect(key, f"orbit.{key}")
