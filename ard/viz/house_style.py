import ard


def get_stylesheets(use_tex=False, dark_background=True):
    house_style_dir = ard.BASE_DIR.parent / "assets" / "house_style"
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
