# FLOWFarm Integration in Ard

This folder contains Ard's Python-Julia integration utilities for FLOWFarm.

## What this integration does

- Boots Julia through JuliaCall.
- Activates Ard's local Julia environment (`julia_env`).
- Loads FLOWFarm and builds farm and sparse structs for Ard components.
- Exposes helper functions used by the component wrapper in `ard/farm_aero/flowfarm.py`.

## Threading behavior

- Ard supports Julia threading through JuliaCall.
- The FLOWFarm update callback used by Ard is implemented in pure Julia (not Python callback) to avoid PythonCall thread-safety crashes.
- If you configure Julia threads with environment variables, set them **before** importing Ard/JuliaCall.

Recommended JuliaCall env options for threaded runs:

- `PYTHON_JULIACALL_THREADS=<N or auto>`
- `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`
- (optional) `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` to avoid nested thread oversubscription

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

- Kernel/process crash when threads > 1:
	- Ensure pure Julia callback path is active (current Ard default).
	- Ensure thread env vars are set before importing Ard.
	- Start with `PYTHON_JULIACALL_THREADS=1`, then increase.
- Julia environment mismatch errors:
	- Re-instantiate the local Julia env in `julia_env`.
	- Confirm FLOWFarm revision/pin is compatible with your Julia runtime.

