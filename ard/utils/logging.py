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

    if not hasattr(component, "_problem_meta"):
        return None
    problem_meta = component._problem_meta

    if "model_ref" not in problem_meta:
        return None
    model = problem_meta["model_ref"]()

    if not hasattr(model, "iter_count"):
        return None
    iter_count = model.iter_count

    return iter_count


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

    logs_dir = [
        "logs",
    ]
    iter = extract_iter(component)
    if iter is not None:
        logs_dir += [f"iter_{iter:04d}"]
    subdir_logger = component.pathname.split(
        "."
    )  # mirror the comp path for a log directory
    dir_reports = Path(
        component._problem_meta["reports_dir"]
    )  # find the reports directory
    path_logfile_template = Path(
        dir_reports.parent,
        *logs_dir,
        *subdir_logger,
        f"%s_rank{component._comm.rank:03d}.txt",
    )  # put the logs directory parallel to it
    path_logfile_stdout = Path(path_logfile_template.as_posix() % "stdout")
    path_logfile_stderr = Path(path_logfile_template.as_posix() % "stderr")

    # make a clean log location for this component
    try:
        path_logfile_stdout.parent.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        shutil.rmtree(path_logfile_stdout.parent, ignore_errors=True)
        path_logfile_stdout.parent.mkdir(parents=True, exist_ok=True)

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
