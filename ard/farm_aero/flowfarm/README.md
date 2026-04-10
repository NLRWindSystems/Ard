# FLOWFarm Integration in Ard

This folder contains Ard's Python-Julia integration utilities for FLOWFarm.

## Julia setup (required before first use)

FLOWFarm runs inside Julia via [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/). You need Julia installed before running any FLOWFarm components. Users who do not use FLOWFarm do not need Julia at all — it is loaded lazily only when a FLOWFarm component is initialized.

### 1. Create a conda environment with Python + juliaup (recommended)

```bash
conda create --name ard-FLOWFarm python=3.13 juliaup
conda activate ard-FLOWFarm
```

Set both Julia depots to a path inside the active conda env:

```bash
conda env config vars set JULIA_DEPOT_PATH=${CONDA_PREFIX}/.julia
conda env config vars set JULIAUP_DEPOT_PATH=${CONDA_PREFIX}/.julia
conda deactivate
conda activate ard-FLOWFarm
```

Install Julia (1.x) using juliaup from that environment:

```bash
juliaup add release
juliaup default release
julia --version
```

Then install Ard (including FLOWFarm dependencies):

```bash
pip install -e ".[dev,flowfarm]"
```

### 2. Pre-generate the Julia environment (optional)

On first use Ard will resolve and instantiate the Julia environment automatically. If you prefer to do this ahead of time — for example on an HPC cluster node without internet access at runtime — run once from your terminal:

```bash
julia --project="<path-to-ard>/ard/farm_aero/flowfarm/julia_env" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"
```

Replace `<path-to-ard>` with the absolute path to the `Ard` directory. This downloads FLOWFarm and its dependencies. It may take several minutes on first run.

## What this integration does

- Boots Julia through JuliaCall.
- Activates Ard's local Julia environment (`julia_env`).
- Loads FLOWFarm and builds farm and sparse structs for Ard components.
- Exposes helper functions used by the component wrapper in `ard/farm_aero/flowfarm/component.py`.

## Threading and parallelism

### OpenMDAO does not multithread

OpenMDAO is single-threaded by design. Its solver loops (Newton, Gauss-Seidel, NLBGS, etc.) are serial. The only parallelism OpenMDAO exposes is MPI-based **process** parallelism via `ParallelGroup`, which spawns separate processes — not threads. This means OpenMDAO will never call the FLOWFarm component from multiple threads simultaneously, so there is no concurrency risk from the OpenMDAO layer.

### Julia internal threading (FLOWFarm parallelism)

Threading in this integration refers to Julia's own thread pool, which FLOWFarm can use internally to parallelize wake calculations across turbines. This is separate from and independent of OpenMDAO.

Julia's thread count is fixed at startup and cannot be changed at runtime. Configure it **before** importing Ard or JuliaCall:

```python
import os
os.environ["PYTHON_JULIACALL_THREADS"] = "4"   # or "auto" to use all cores
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
```

For threaded runs on shared-memory machines, also consider limiting BLAS and OpenMP thread pools to avoid nested oversubscription (Julia threads × BLAS threads × cores):

```python
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
```

These must be set before the first `import ard` call in your script or notebook.

### Why a pure Julia callback

The FLOWFarm update callback in Ard is implemented entirely in Julia (not as a Python callable passed into Julia). This is required for thread safety: JuliaCall does not support calling back into Python from Julia threads other than the main thread. Using a pure Julia callback avoids this restriction and allows FLOWFarm to use all available Julia threads.

## Tolerance behavior

- FLOWFarm sparse-structure tolerance uses `modeling_options.flowfarm.tolerance`.
- If not provided, the default is `1e-16`.

Example:

```yaml
modeling_options:
	flowfarm:
		tolerance: 1.0e-16
```

## Key files

- `_jl_bootstrap.py`: Julia runtime bootstrap and env activation helpers.
- `flowfarm_model.py`: FLOWFarm model-construction utilities and option validation.
- `component.py`: OpenMDAO component wrapper that uses this integration.

## Troubleshooting

### Julia manifest warnings on first run

If you see warnings like "manifest resolved with a different julia version" or "project dependencies have changed since the manifest was last resolved", it means the local `Manifest.toml` is missing or stale. Ard will attempt to rebuild it automatically. If it does not, run:

```bash
julia --project="<path-to-ard>/ard/farm_aero/flowfarm/julia_env" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"
```

Then restart your Jupyter kernel. The `Manifest.toml` is not committed to the repository — it is always generated locally for your Julia version.


This comes from your **global** Julia environment, not Ard's. JuliaCall triggers the IPython/Jupyter juliacall extension on import.

### Wrong Julia version being used

If Julia 1.11+ is picked up instead of 1.10, check your `PATH`. `juliaup default 1.10` sets the default for commands run via juliaup, but if `/opt/homebrew/bin/julia` or another system Julia takes precedence in your shell, JuliaCall may use that instead.

### Kernel/process crash when threads > 1

- Ensure pure Julia callback path is active (current Ard default).
- Ensure thread env vars are set before importing Ard.
- Start with `PYTHON_JULIACALL_THREADS=1`, then increase.