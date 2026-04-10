import sys

from ard.farm_aero.flowfarm import flowfarm_model as _flowfarm_model

sys.modules[__name__] = _flowfarm_model
