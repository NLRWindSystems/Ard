# ard/farm_aero/flowfarm/__init__.py
from .flowfarm_model import FlowFarmModel
from ._jl_bootstrap import ensure_flowfarm_loaded, get_julia_module, get_julia_runtime

__all__ = [
    "FlowFarmModel",
    "ensure_flowfarm_loaded",
    "get_julia_module",
    "get_julia_runtime",
]
