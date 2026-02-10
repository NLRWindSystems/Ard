import numpy as np

import openmdao.api as om


class CollectionTemplate(om.ExplicitComponent):
    """
    Template component for a energy collection system.

    A energy collection system component, based on this template, will compute the
    energy collection system necessary for a farm given its layout, turbine
    definitions, and substation location.

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
    ----------------
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
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of OM component."""
        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.windIO_plant = self.modeling_options["windIO_plant"]
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]
        self.N_substations = self.modeling_options["layout"]["N_substations"]
        if "x_turbines" in self.modeling_options["layout"]:
            self.x_turbines = self.modeling_options["layout"]["x_turbines"]
        else:
            self.x_turbines = (
                self.windIO_plant.get("wind_farm", {})
                .get("layouts", {})
                .get("coordinates", {})
                .get("x", np.zeros((self.N_turbines,)))
            )
        if "y_turbines" in self.modeling_options["layout"]:
            self.y_turbines = self.modeling_options["layout"]["y_turbines"]
        else:
            self.y_turbines = (
                self.windIO_plant.get("wind_farm", {})
                .get("layouts", {})
                .get("coordinates", {})
                .get("y", np.zeros((self.N_turbines,)))
            )
        self.x_substations = np.array(
            [
                substation["electrical_substation"]["coordinates"]["x"]
                for substation in self.windIO_plant["wind_farm"][
                    "electrical_substations"
                ]
            ]
        )
        self.y_substations = np.array(
            [
                substation["electrical_substation"]["coordinates"]["y"]
                for substation in self.windIO_plant["wind_farm"][
                    "electrical_substations"
                ]
            ]
        )

        # set up inputs for farm layout
        self.add_input("x_turbines", self.x_turbines, units="m")
        self.add_input("y_turbines", self.y_turbines, units="m")
        self.add_input("x_substations", self.x_substations, units="m")
        self.add_input("y_substations", self.y_substations, units="m")
        self.add_discrete_input("x_border", None)
        self.add_discrete_input("y_border", None)

        # set up outputs for the collection system
        self.add_output("total_length_cables", 0.0, units="m")
        self.add_output("terse_links", np.full((self.N_turbines,), -1))
        self.add_discrete_output("length_cables", np.zeros((self.N_turbines,)))
        self.add_discrete_output("load_cables", np.zeros((self.N_turbines,)))
        self.add_discrete_output("max_load_cables", 0.0)
        self.add_discrete_output("graph", None)

    def compute(
        self,
        inputs,
        outputs,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        ###########################################
        #                                         #
        # IMPLEMENT THE COLLECTION COMPONENT HERE #
        #                                         #
        ###########################################

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )
