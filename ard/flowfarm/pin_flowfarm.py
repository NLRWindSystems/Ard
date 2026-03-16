# ard/farm_aero/flowfarm/pin_flowfarm.py
from __future__ import annotations
import sys, pathlib
import juliacall
from juliacall import (
    Pkg as jlPkg,
)  # JuliaPkg via JuliaCall (documented) [1](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)

FLOWFARM_GIT_URL = "https://github.com/byuflowlab/FLOWFarm.jl"
FLOWFARM_REV = "typestability"  # <-- BRANCH PIN


def main(argv=None):
    env_dir = pathlib.Path(__file__).parent / "julia_env"
    print(f"[pin] Activating: {env_dir}")
    jlPkg.activate(str(env_dir))
    print("[pin] Instantiating (may download packages on first run)…")
    jlPkg.instantiate()  # creates/updates Manifest.toml [1](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)

    # If FLOWFarm exists with a different source/UUID, replace it with our pin.
    try:
        jlPkg.rm("FLOWFarm")
    except Exception:
        pass

    print(f"[pin] Pkg.add url={FLOWFARM_GIT_URL} rev={FLOWFARM_REV}")
    jlPkg.add(
        url=FLOWFARM_GIT_URL, rev=FLOWFARM_REV
    )  # captures exact revision in Manifest [1](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)

    jl = juliacall.newmodule("ArdFLOWFarmPin")
    print("[pin] Loading FLOWFarm…")
    jl.seval(
        "using FLOWFarm"
    )  # FLOWFarm usage/install documented in repo [2](https://github.com/byuflowlab/FlowFarm.jl)

    # Optional: precompile to warm caches on first run
    if "--precompile" in (argv or []):
        print("[pin] Precompiling Julia environment…")
        jlPkg.precompile()

    manifest = env_dir / "Manifest.toml"
    print(f"[done] Manifest at: {manifest if manifest.exists() else '(missing)'}")


if __name__ == "__main__":
    main(sys.argv[1:])
