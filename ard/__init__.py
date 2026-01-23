from . import collection
from . import layout
from . import offshore
from . import wind_query
from . import utils

from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent
ASSET_DIR = BASE_DIR / "api" / "default_systems"


def get_house_style(use_tex=False, dark_background=True):
    house_style_dir = BASE_DIR.parent / "assets" / "house_style"
    ard_stylesheet = (
        house_style_dir / f"stylesheet_ard{"" if use_tex else "_notex"}.mplstyle"
    )
    nrel_stylesheet = house_style_dir / f"stylesheet_nrel.mplstyle"
    styles = []
    if dark_background:
        styles.append("dark_background")
    styles.append(ard_stylesheet.as_uri())
    styles.append(nrel_stylesheet.as_uri())
    return styles
