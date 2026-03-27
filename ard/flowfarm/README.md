# FLOWFarm Integration in Ard

This folder contains Ard's Python-Julia integration utilities for FLOWFarm.

## Julia setup (required before first use)

FLOWFarm runs inside Julia via [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/). You need Julia installed before running any FLOWFarm components. Users who do not use FLOWFarm do not need Julia at all — it is loaded lazily only when a FLOWFarm component is initialized.

### 1. Install Julia via juliaup (recommended)

[juliaup](https://github.com/JuliaLang/juliaup) is the official Julia version manager. Install it with:

```bash
curl -fsSL https://install.julialang.org | sh
```

Install any recent stable Julia release (1.10 or later):

```bash
juliaup add release
juliaup default release
```

Verify:

```bash
julia --version
```

Ard's Julia environment has no hard version pin. The `Manifest.toml` is not committed to the repository — it is generated locally the first time you run a FLOWFarm component, so it will always match your installed Julia version. If you need to generate it ahead of time (e.g. on a cluster before a job runs), see step 2.

### 2. Pre-generate the Julia environment (optional)

On first use Ard will resolve and instantiate the Julia environment automatically. If you prefer to do this ahead of time — for example on an HPC cluster node without internet access at runtime — run once from your terminal:

```bash
julia --project="<path-to-ard>/ard/flowfarm/julia_env" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"
```

Replace `<path-to-ard>` with the absolute path to the `Ard` directory. This downloads FLOWFarm and its dependencies. It may take several minutes on first run.

### 3. Install the JuliaCall Python package

```bash
pip install juliacall
```

`juliacall` is not listed in Ard's core dependencies because it is only needed for FLOWFarm. Install it separately before using FLOWFarm components.

## What this integration does

- Boots Julia through JuliaCall.
- Activates Ard's local Julia environment (`julia_env`).
- Loads FLOWFarm and builds farm and sparse structs for Ard components.
- Exposes helper functions used by the component wrapper in `ard/farm_aero/flowfarm.py`.

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
- `../ard/farm_aero/flowfarm.py`: OpenMDAO component wrapper that uses this integration.

## Troubleshooting

### Julia manifest warnings on first run

If you see warnings like "manifest resolved with a different julia version" or "project dependencies have changed since the manifest was last resolved", it means the local `Manifest.toml` is missing or stale. Ard will attempt to rebuild it automatically. If it does not, run:

```bash
julia --project="<path-to-ard>/ard/flowfarm/julia_env" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"
```

Then restart your Jupyter kernel. The `Manifest.toml` is not committed to the repository — it is always generated locally for your Julia version.

### Revise / DistributedExt error in Jupyter

```
Error during loading of extension DistributedExt of Revise
```

This comes from your **global** Julia environment, not Ard's. JuliaCall triggers the IPython/Jupyter juliacall extension on import, which loads Revise from your global env. Fix it by running:

```bash
julia -e "using Pkg; Pkg.add(\"Distributed\"); Pkg.resolve()"
```

### Wrong Julia version being used

If Julia 1.11+ is picked up instead of 1.10, check your `PATH`. `juliaup default 1.10` sets the default for commands run via juliaup, but if `/opt/homebrew/bin/julia` or another system Julia takes precedence in your shell, JuliaCall may use that instead.

To force a specific version for a notebook session, add this to the **first cell** before any other imports:

```python
import os
os.environ["PYTHON_JULIACALL_EXE"] = "julia +1.10"
os.environ["ARD_FLOWFARM_RESPECT_JULIACALL_ENV"] = "1"
```

`ARD_FLOWFARM_RESPECT_JULIACALL_ENV=1` is required — without it Ard's bootstrap strips the override.

### Kernel/process crash when threads > 1

- Ensure pure Julia callback path is active (current Ard default).
- Ensure thread env vars are set before importing Ard.
- Start with `PYTHON_JULIACALL_THREADS=1`, then increase.

