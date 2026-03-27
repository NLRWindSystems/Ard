import shutil
import warnings

import pytest


def _julia_available() -> bool:
    """Return True only if the julia executable and juliacall package are present.

    Deliberately avoids importing juliacall here — that would start Julia at
    collection time, which is expensive and can cause hangs or errors.
    """
    if shutil.which("julia") is None:
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("juliacall") is not None
    except Exception:
        return False


JULIA_AVAILABLE = _julia_available()


def pytest_collection_modifyitems(config, items):
    """Auto-skip julia-marked tests when Julia is not installed, and print a note."""
    if JULIA_AVAILABLE:
        return

    skip = pytest.mark.skip(
        reason=(
            "Julia not installed — install Julia via juliaup and "
            "`pip install .[flowfarm]` to run FLOWFarm tests"
        )
    )
    julia_tests = [item for item in items if "julia" in item.keywords]

    if julia_tests:
        warnings.warn(
            f"\nARD NOTE: Julia not found — {len(julia_tests)} FLOWFarm integration "
            "test(s) will be skipped.\n"
            "  To enable: install Julia (juliaup) and run `pip install .[flowfarm]`\n",
            UserWarning,
            stacklevel=2,
        )
        for item in julia_tests:
            item.add_marker(skip)
