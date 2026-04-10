import sys

from ard.farm_aero.flowfarm import _jl_bootstrap as _bootstrap

sys.modules[__name__] = _bootstrap
