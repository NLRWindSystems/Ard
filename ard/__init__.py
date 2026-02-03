from pathlib import Path

from . import collection
from . import layout
from . import offshore
from . import wind_query
from . import utils
from . import viz

from .viz import house_style

BASE_DIR = Path(__file__).parent.absolute()
ASSET_DIR = BASE_DIR / "api" / "default_systems"
