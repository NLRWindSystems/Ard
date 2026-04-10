# ard/farm_aero/flowfarm/_jl_bootstrap.py
from __future__ import annotations
import pathlib

_jl_runtime = None
_flowfarm_env_initialized = False


def get_julia_runtime():
    """Return (Main, Pkg) from JuliaCall with Ard-safe bootstrap behavior."""
    global _jl_runtime
    if _jl_runtime is not None:
        return _jl_runtime

    from juliacall import Main as jl_main
    from juliacall import Pkg as jl_pkg

    _jl_runtime = (jl_main, jl_pkg)
    return _jl_runtime


def ensure_flowfarm_loaded():
    """Activate Ard Julia env and load FLOWFarm in Julia Main."""
    global _flowfarm_env_initialized
    jl_main, jl_pkg = get_julia_runtime()
    if not _flowfarm_env_initialized:
        env_dir = pathlib.Path(__file__).parent / "julia_env"
        jl_pkg.activate(str(env_dir))
        jl_pkg.instantiate()
        _flowfarm_env_initialized = True

    if "FLOWFarm" not in dir(jl_main):
        jl_main.seval("using FLOWFarm")
    return jl_main
