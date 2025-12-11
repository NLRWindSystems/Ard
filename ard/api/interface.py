import importlib
import openmdao.api as om
from openmdao.drivers.doe_driver import DOEGenerator
from ard.utils.io import load_yaml, replace_key_value
from ard.cost.wisdem_wrap import (
    LandBOSSE_setup_latents,
    ORBIT_setup_latents,
    FinanceSE_setup_latents,
)
import windIO
from ard import ASSET_DIR
from typing import Union


def set_up_ard_model(input_dict: Union[str, dict], root_data_path: str = None):
    """
    Set up an OpenMDAO model for Ard based on the provided input dictionary or YAML file.

    This function initializes and configures an OpenMDAO problem using a system
    specification, modeling options, and analysis options. It supports default
    system configurations (e.g., "onshore", "offshore_floating") and allows for
    recursive setup of subsystems and connections.

    Parameters
    ----------
    input_dict : Union[str, dict]
        A dictionary or a path to a YAML file containing the configuration for the Ard model.
        The dictionary or YAML file must include:

        - "system" : str or dict
            The name of the default system to use (e.g., "onshore") or a custom system specification.
        - "modeling_options" : dict
            A dictionary defining the modeling options for the system (e.g., turbine specs, farm layout).
        - "analysis_options" : dict
            A dictionary defining the analysis options, including driver settings, design variables,
            constraints, objectives, and recorder configuration.

    root_data_path : str, optional
        The root path for resolving relative paths in the system configuration. Defaults to None.

    Returns
    -------
    om.Problem
        An OpenMDAO problem instance with the defined system hierarchy, modeling options,
        and analysis options.

    Raises
    ------
    ValueError
        If an invalid default system is specified or if required keys are missing in the input dictionary.

    Notes
    -----
    - The function uses `set_up_system_recursive` to recursively build the system hierarchy.
    - Latent variables for LandBOSSE, ORBIT, and FinanceSE are automatically set up if their
      respective components are present in the model.

    """

    # load dictionary if string is given
    if isinstance(input_dict, str):
        input_dict, root_data_path = load_yaml(input_dict, return_path=True)

    # load default system if requested and available
    available_default_systems = [
        "onshore",
        "onshore_batch",
        "onshore_no_cable_design",
        "offshore_monopile",
        "offshore_monopile_no_cable_design",
        "offshore_floating",
        "offshore_floating_no_cable_design",
    ]

    if isinstance(input_dict["system"], str):
        if input_dict["system"] in available_default_systems:
            system = load_yaml(ASSET_DIR / f"ard_system_{input_dict['system']}.yaml")

            input_dict["system"] = replace_key_value(
                target_dict=system,
                target_key="modeling_options",
                new_value=input_dict["modeling_options"],
            )
        else:
            raise (
                ValueError(
                    f"invalid default system '{input_dict['system']}' specified. Must be one of {available_default_systems}"
                )
            )

    # replace empty data_path specs
    input_dict["system"] = replace_key_value(
        target_dict=input_dict["system"],
        target_key="data_path",
        new_value=root_data_path,
        replace_none_only=True,
    )

    # validate windIO dictionary
    windIO_dict = input_dict["modeling_options"]["windIO_plant"]
    windIO.validate(windIO_dict, schema_type="plant/wind_energy_system")

    # set up the openmdao problem
    prob = set_up_system_recursive(
        input_dict=input_dict["system"],
        modeling_options=input_dict["modeling_options"],
        analysis_options=input_dict["analysis_options"],
    )

    return prob


def set_up_system_recursive(
    input_dict: dict,
    system_name: str = "top_level",
    case_name: str | None = None,
    work_dir: str = "case_files",
    parent_group=None,
    modeling_options: dict = None,
    analysis_options: dict = None,
    _depth: int = 0,
):
    """
    Recursively sets up an OpenMDAO system based on the input dictionary.

    Args:
        input_dict (dict): Dictionary defining the system hierarchy.
        parent_group (om.Group, optional): The parent group to which subsystems are added.
                                           Defaults to None, which initializes the top-level problem.

    Returns:
        om.Problem: The OpenMDAO problem with the defined system hierarchy.
    """

    # grab the case name if it's supplied in the system yaml
    if case_name is None:
        case_name = modeling_options.get(
            "case_name",
            input_dict.get("modeling_options", {}).get(
                "case_name",
                "ard_problem",
            ),
        )

    # Initialize the top-level problem if no parent group is provided
    if parent_group is None:
        prob = om.Problem(
            name=case_name,
            work_dir=work_dir,
        )
        parent_group = prob.model
        # parent_group.name = "ard_model"
    else:
        prob = None

    # Add subsystems directly from the input dictionary
    if hasattr(parent_group, "name") and (parent_group.name != ""):
        print(f"Adding {system_name} to {parent_group.name}")
    else:
        print(f"Adding {system_name}")
    if "systems" in input_dict:  # Recursively add nested subsystems]
        if _depth > 0:
            group = parent_group.add_subsystem(
                name=system_name,
                subsys=om.Group(),
                promotes=input_dict.get("promotes", None),
            )
        else:
            group = parent_group
        for subsystem_key, subsystem_data in input_dict["systems"].items():
            set_up_system_recursive(
                subsystem_data,
                parent_group=group,
                system_name=subsystem_key,
                modeling_options=modeling_options,
                analysis_options=None,
                _depth=_depth + 1,
            )
        if "approx_totals" in input_dict:
            print(f"\tActivating approximate totals on {system_name}")
            group.approx_totals(**input_dict["approx_totals"])

    else:
        subsystem_data = input_dict

        if "object" not in subsystem_data:
            raise ValueError(f"Ard subsystem '{system_name}' missing 'object' spec.")
        if "promotes" not in subsystem_data:
            raise ValueError(f"Ard subsystem '{system_name}' missing 'promotes' spec.")

        # Dynamically import the module and get the subsystem class
        Module = importlib.import_module(subsystem_data["module"])
        SubSystem = getattr(Module, subsystem_data["object"])

        # Convert specific promotes to tuples
        promotes = [
            tuple(p) if isinstance(p, list) else p for p in subsystem_data["promotes"]
        ]

        # Add the subsystem to the parent group with kwargs
        parent_group.add_subsystem(
            name=system_name,
            subsys=SubSystem(**subsystem_data.get("kwargs", {})),
            promotes=promotes,
        )

    # Handle connections within the parent group
    if "connections" in input_dict:
        for connection in input_dict["connections"]:
            src, tgt = connection  # Unpack the connection as [src, tgt]
            parent_group.connect(src, tgt)

    # Set up the problem if this is the top-level call
    if prob is not None:

        if analysis_options:
            # set up driver
            if "driver" in analysis_options:
                Driver = getattr(om, analysis_options["driver"]["name"])

                # handle DOE drivers with special treatment
                if Driver == om.DOEDriver:
                    generator = None
                    if "generator" in analysis_options["driver"]:
                        if type(analysis_options["driver"]["generator"]) == dict:
                            gen_dict = analysis_options["driver"]["generator"]
                            generator = getattr(om, gen_dict["name"])(
                                **gen_dict["args"]
                            )
                        elif isinstance(
                            analysis_options["driver"]["generator"], DOEGenerator
                        ):
                            generator = analysis_options["driver"]["generator"]
                        else:
                            raise NotImplementedError(
                                "Only dictionary-specified or OpenMDAO "
                                "DOEGenerator generators have been implemented."
                            )
                    prob.driver = Driver(generator)
                else:
                    prob.driver = Driver()

                # handle the options now
                if "options" in analysis_options["driver"]:
                    for option, value_driver_option in analysis_options["driver"][
                        "options"
                    ].items():
                        if option == "opt_settings":
                            for (
                                key_opt_setting,
                                value_opt_setting,
                            ) in value_driver_option.items():
                                prob.driver.opt_settings[key_opt_setting] = (
                                    value_opt_setting
                                )
                        else:
                            prob.driver.options[option] = value_driver_option

                    # set design variables
            if "design_variables" in analysis_options:
                for var_name, var_data in analysis_options["design_variables"].items():
                    prob.model.add_design_var(var_name, **var_data)

            # set constraints
            if "constraints" in analysis_options:
                for constraint_name, constraint_data in analysis_options[
                    "constraints"
                ].items():
                    prob.model.add_constraint(constraint_name, **constraint_data)

            # set objective
            if "objectives" in analysis_options:
                for obj_name, obj_options in analysis_options["objectives"].items():
                    obj_options = {} if (obj_options is None) else obj_options
                    prob.model.add_objective(
                        obj_name,
                        **obj_options,
                    )

            # Set up the recorder if specified in the input dictionary
            if "recorder" in analysis_options:
                recorder_filepath = analysis_options["recorder"].get("filepath")
                if recorder_filepath:
                    recorder = om.SqliteRecorder(recorder_filepath)
                    prob.add_recorder(recorder)
                    prob.driver.add_recorder(recorder)

        # TODO! THIS IS NECESSARY FOR SOME REASON WHEN RUNNING FREE
        # OPTIMIZATIONS. THIS SHOULDN'T BE NEEDED...
        prob.model.set_input_defaults(
            "x_turbines",
            # input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"]["coordinates"]["x"],
            units="m",
        )
        prob.model.set_input_defaults(
            "y_turbines",
            # input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"]["coordinates"]["y"],
            units="m",
        )

        # setup the openmdao problem
        prob.setup()

    return prob
