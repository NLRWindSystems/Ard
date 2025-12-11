from contextlib import redirect_stdout, redirect_stderr
from functools import wraps
from pathlib import Path
import shutil

import openmdao.core.component


def extract_iter(component):
    """
    Extract the iter_count iff it exists, otherwise return None

    Extract the iteration count from a component's associated model.
    This function attempts to retrieve the iteration count from a component by
    traversing through its problem metadata and model reference. It safely
    handles cases where any of the required attributes or keys don't exist.

    Parameters
    ----------
        component: An object that may contain a _problem_meta attribute with
                  model reference information.
    Returns
    -------
        int or None: The iteration count from the model if it exists and is
                    accessible, otherwise None.
            The function returns None in the following cases:
            - component doesn't have a _problem_meta attribute
            - problem_meta doesn't contain a "model_ref" key
            - the model doesn't have an iter_count attribute
    """

    # extract the iter count if it exists, returning and handling none otherwise
    problem_meta = getattr(component, "_problem_meta", None)
    model = problem_meta.get("model_ref", lambda _: None)()
    iter_count = getattr(model, "iter_count", None)

    return iter_count


def get_storage_directory(
    component,
    storage_type: str = "logs",
    get_iter: bool = False,
    clean: bool = False,
):
    """
    Get a storage directory for the component constructed here.

    Take a component and create a storage directory (for, e.g. logs or init
    files), mirroring the OpenMDAO model structure as subdirectories, returning
    a pathlib.Path to the storage directory.

    Parameters
    ----------
    component : openmdao.core.Component
        an OpenMDAO component for which we want to create a storage directory
    storage_type : str, optional
        the type of storage sub-directory to make, by default "logs"
    get_iter : bool, optional
        should the storage directory tree be given an iteration subdirectory, by
        default False
    clean : bool, optional
        should the directory tree, if it already exists, be cleaned out, by
        default False

    Returns
    -------
    pathlib.Path
        the path to the storage subdirectory created
    """
    # the storage type we're doing (logs, discipline scripts, etc.)
    storage_dir = [
        storage_type,
    ]
    # if there's an iteration number to grab, grab it and add it to the dir
    iter = extract_iter(component) if get_iter else None
    if iter:
        storage_dir += [f"iter_{iter:04d}"]
    # mirror the comp path for a log directory
    subdir_logger = component.pathname.split(".")
    # find the reports directory
    dir_reports = Path(component._problem_meta["reports_dir"])
    # put the storage directory next to it
    path_storage = Path(dir_reports.parent, *storage_dir, *subdir_logger)

    # make a clean log location for this component if permitted
    try:
        path_storage.mkdir(parents=True, exist_ok=False)
    except FileExistsError:  # handle a FileExists, but raise anything else
        if clean:
            shutil.rmtree(path_storage, ignore_errors=True)
            path_storage.mkdir(parents=True, exist_ok=True)
        else:
            raise

    return path_storage


def name_create_log(component, iter: int = None):
    """
    For a given component, clean and create component- and rank-unique logfiles.

    Take a component and create logs, parallel to the reports file, mirroring
    the OpenMDAO model structure with stdout and stderr files for each rank,
    and finally return the file paths for the component to redirect stdout and
    stderr to.

    Parameters
    ----------
    component : openmdao.core.component.Component
        An OpenMDAO component that we want to capture stdout/stderr for

    Returns
    -------
    pathlib.Path
        a path to in the log system to dump stdout to
    pathlib.Path
        a path to in the log system to dump err to
    """

    # make sure we are dealing with an OM component
    if not isinstance(component, openmdao.core.component.Component):
        raise TypeError(
            f"Expected openmdao.core.component.Component, got {type(component)}"
        )

    path_logfile_template = (
        get_storage_directory(component, "logs", True, clean=True)
        / f"%s_rank{component._comm.rank:03d}.txt"
    )
    path_logfile_stdout = Path(path_logfile_template.as_posix() % "stdout")
    path_logfile_stderr = Path(path_logfile_template.as_posix() % "stderr")

    # return stdout and stderr files
    return path_logfile_stdout.absolute(), path_logfile_stderr.absolute()


def component_log_capture(compute_func, iter: int = None):
    """
    Decorator that redirects stdout and stderr to component-wise and rank-wise logfiles.

    This decorator will redirect stdout and stderr to component-wise and
    rank-wise logfiles, which are determined by the `name_create_log` function.
    The decorator uses context managers to redirect output streams to these
    files, ensuring that all print statements and errors within the function are
    logged appropriately.

    func : Callable
        The function to be decorated. It should be a method of a class, as
        `self` is expected as the first argument.

    Callable
        The wrapped function with stdout and stderr redirected to log files
        during its execution.
    """

    @wraps(compute_func)
    def wrapper(self, *args, **kwargs):

        # extract from modeling options the stdio_capture option iff it exists
        stdio_capture = getattr(self, "modeling_options", {}).get("stdio_capture")
        # bail out, returning the function w/ no changes if it doesn't
        if not stdio_capture:
            return compute_func(self, *args, **kwargs)

        # if we get here, we want to capture stdio

        # get log file paths
        path_stdout_log, path_stderr_log = name_create_log(self)

        try:
            # use context manager to redirect stdout & stderr
            with (
                open(path_stdout_log, "a") as stdout_file,
                open(path_stderr_log, "a") as stderr_file,
                redirect_stdout(stdout_file),
                redirect_stderr(stderr_file),
            ):
                return compute_func(self, *args, **kwargs)
        except Exception:
            raise  # make sure the exception is raised

    return wrapper
