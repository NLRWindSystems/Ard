from contextlib import redirect_stdout, redirect_stderr
from functools import wraps
from pathlib import Path
import shutil

import openmdao.core.component


def name_create_log(component, iter: int = None):
    """
    for a given component, clean and create component- and rank-unique logfiles

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
    assert isinstance(component, openmdao.core.component.Component)

    logs_dir = ["logs"]
    if iter is not None:
        logs_dir += f"iter_{iter:04d}"
    subdir_logger = component.pathname.split(
        "."
    )  # mirror the comp path for a log directory
    dir_reports = Path(
        component._problem_meta["reports_dir"]
    )  # find the reports directory
    path_logfile_template = Path(
        dir_reports.parent,
        "logs",
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
    return path_logfile_stdout, path_logfile_stderr


def component_log_capture(compute_func, iter: int = None):
    """
    decorator that redirects stdout and stderr to disciplinary rankwise logfiles

    This decorator will redirects stdout and stderr to discipline-wise and
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

        # use context manager to redirect stdout & stderr
        with (
            open(path_stdout_log, "a") as stdout_file,
            open(path_stderr_log, "a") as stderr_file,
            redirect_stdout(stdout_file),
            redirect_stderr(stderr_file),
        ):
            return compute_func(self, *args, **kwargs)

    return wrapper
