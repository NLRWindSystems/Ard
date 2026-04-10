from __future__ import annotations

import pathlib
import warnings
import numpy as np

from ._jl_bootstrap import ensure_flowfarm_loaded, get_julia_runtime

# ------------------------------------------------------------------------------
# Configuration (project activation)
# ------------------------------------------------------------------------------

# If you're using the embedded project (recommended), point to it explicitly:
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_JULIA_PROJECT_DIR = _THIS_DIR / "julia_env"


def _get_jl_main():
    jl, _ = get_julia_runtime()
    return jl


def _ensure_env_activated():
    # Prefer explicit activation over relying on JULIA_PROJECT env var.
    _, jl_pkg = get_julia_runtime()
    jl_pkg.activate(str(_JULIA_PROJECT_DIR))
    jl_pkg.instantiate()  # ensures Manifest is honored / deps are present


def _ensure_flowfarm_loaded():
    ensure_flowfarm_loaded()


# ------------------------------------------------------------------------------
# Utility: Julia Vector conversion (optional; JuliaCall already converts NumPy arrays)
# ------------------------------------------------------------------------------


def _jvec(x):
    """Convert Python list/array → Julia Vector{Float64} (explicit)."""
    jl = _get_jl_main()
    return jl.Vector[jl.Float64](list(map(float, np.asarray(x).ravel())))


def _resolve_flowfarm_constructor(flowfarm_module, candidate_names):
    """Return the first FLOWFarm constructor that exists from candidate names."""
    for name in candidate_names:
        if hasattr(flowfarm_module, name):
            return getattr(flowfarm_module, name)
    return None


def _build_flowfarm_power_model(
    flowfarm_module,
    has_cp_curve,
    cp_curve,
    constant_cp,
    fallback_wind_speeds,
):
    """Build a FLOWFarm power model from Cp curve or constant Cp fallback."""
    if has_cp_curve:
        power_points_ctor = _resolve_flowfarm_constructor(
            flowfarm_module,
            ["PowerModelCpPoints"],
        )
        if power_points_ctor is None:
            raise AttributeError(
                "FLOWFarm.PowerModelCpPoints constructor was not found."
            )
        return power_points_ctor(
            _jvec(cp_curve["Cp_wind_speeds"]),
            _jvec(cp_curve["Cp_values"]),
        )

    power_constant_ctor = _resolve_flowfarm_constructor(
        flowfarm_module,
        ["PowerModelConstantCp", "PowerModelCpConstant"],
    )
    if power_constant_ctor is not None:
        return power_constant_ctor(float(constant_cp))

    # Last-resort fallback if constant-Cp constructor name differs by FLOWFarm version.
    # Approximate a constant Cp model using points at representative wind speeds.
    warnings.warn(
        "FLOWFarm constant-Cp constructor not found; falling back to PowerModelCpPoints with constant Cp.",
        UserWarning,
        stacklevel=2,
    )
    power_points_ctor = _resolve_flowfarm_constructor(
        flowfarm_module,
        ["PowerModelCpPoints"],
    )
    if power_points_ctor is None:
        raise AttributeError("FLOWFarm.PowerModelCpPoints constructor was not found.")
    cp_values = [float(constant_cp)] * len(fallback_wind_speeds)
    return power_points_ctor(_jvec(fallback_wind_speeds), _jvec(cp_values))


def _build_flowfarm_ct_model(
    flowfarm_module,
    has_ct_curve,
    ct_curve,
    constant_ct,
    fallback_wind_speeds,
):
    """Build a FLOWFarm thrust model from Ct curve or constant Ct fallback."""
    if has_ct_curve:
        ct_points_ctor = _resolve_flowfarm_constructor(
            flowfarm_module,
            ["ThrustModelCtPoints"],
        )
        if ct_points_ctor is None:
            raise AttributeError(
                "FLOWFarm.ThrustModelCtPoints constructor was not found."
            )
        return ct_points_ctor(
            _jvec(ct_curve["Ct_wind_speeds"]),
            _jvec(ct_curve["Ct_values"]),
        )

    ct_constant_ctor = _resolve_flowfarm_constructor(
        flowfarm_module,
        ["ThrustModelConstantCt", "ThrustModelCtConstant"],
    )
    if ct_constant_ctor is not None:
        return ct_constant_ctor(float(constant_ct))

    # Last-resort fallback if constant-Ct constructor name differs by FLOWFarm version.
    warnings.warn(
        "FLOWFarm constant-Ct constructor not found; falling back to ThrustModelCtPoints with constant Ct.",
        UserWarning,
        stacklevel=2,
    )
    ct_points_ctor = _resolve_flowfarm_constructor(
        flowfarm_module,
        ["ThrustModelCtPoints"],
    )
    if ct_points_ctor is None:
        raise AttributeError("FLOWFarm.ThrustModelCtPoints constructor was not found.")
    ct_values = [float(constant_ct)] * len(fallback_wind_speeds)
    return ct_points_ctor(_jvec(fallback_wind_speeds), _jvec(ct_values))


def resolve_turbine_inputs_for_flowfarm(windio_turbine):
    """Validate turbine inputs and return a normalized config dict for FLOWFarm."""
    _ensure_flowfarm_loaded()
    jl = _get_jl_main()
    flowfarm_module = jl.FLOWFarm

    scalar_defaults = {
        "generator_efficiency": 1.0,
        "rated_power": 1e6,
        "rated_wind_speed": 10.0,
        "cutin_wind_speed": 0.0,
        "cutout_wind_speed": 100.0,
    }

    missing_scalars = [
        key
        for key in scalar_defaults
        if key not in windio_turbine or windio_turbine[key] is None
    ]
    if missing_scalars:
        defaults_used = {key: scalar_defaults[key] for key in missing_scalars}
        warnings.warn(
            f"FLOWFarm missing turbine inputs {missing_scalars}; using defaults {defaults_used}.",
            UserWarning,
            stacklevel=2,
        )

    performance = windio_turbine.get("performance", {})
    ct_curve = performance.get("Ct_curve", {})
    cp_curve = performance.get("Cp_curve", {})

    has_ct_curve = (
        "Ct_wind_speeds" in ct_curve
        and "Ct_values" in ct_curve
        and ct_curve["Ct_wind_speeds"] is not None
        and ct_curve["Ct_values"] is not None
    )
    has_cp_curve = (
        "Cp_wind_speeds" in cp_curve
        and "Cp_values" in cp_curve
        and cp_curve["Cp_wind_speeds"] is not None
        and cp_curve["Cp_values"] is not None
    )

    constant_ct = performance.get("Ct", performance.get("ct", 0.8))
    constant_cp = performance.get("Cp", performance.get("cp", 0.45))

    if not has_ct_curve:
        warnings.warn(
            f"FLOWFarm missing turbine.performance.Ct_curve; using constant Ct={constant_ct}.",
            UserWarning,
            stacklevel=2,
        )
    if not has_cp_curve:
        warnings.warn(
            f"FLOWFarm missing turbine.performance.Cp_curve; using constant Cp={constant_cp}.",
            UserWarning,
            stacklevel=2,
        )

    fallback_wind_speeds = [
        float(
            windio_turbine.get("cutin_wind_speed", scalar_defaults["cutin_wind_speed"])
        ),
        float(
            windio_turbine.get("rated_wind_speed", scalar_defaults["rated_wind_speed"])
        ),
        float(
            windio_turbine.get(
                "cutout_wind_speed", scalar_defaults["cutout_wind_speed"]
            )
        ),
    ]

    power_model = _build_flowfarm_power_model(
        flowfarm_module,
        has_cp_curve,
        cp_curve,
        constant_cp,
        fallback_wind_speeds,
    )
    ct_model = _build_flowfarm_ct_model(
        flowfarm_module,
        has_ct_curve,
        ct_curve,
        constant_ct,
        fallback_wind_speeds,
    )

    return {
        "generator_efficiency": windio_turbine.get(
            "generator_efficiency", scalar_defaults["generator_efficiency"]
        ),
        "rated_power": windio_turbine.get(
            "rated_power", scalar_defaults["rated_power"]
        ),
        "rated_wind_speed": windio_turbine.get(
            "rated_wind_speed", scalar_defaults["rated_wind_speed"]
        ),
        "cutin_wind_speed": windio_turbine.get(
            "cutin_wind_speed", scalar_defaults["cutin_wind_speed"]
        ),
        "cutout_wind_speed": windio_turbine.get(
            "cutout_wind_speed", scalar_defaults["cutout_wind_speed"]
        ),
        "ct_model": ct_model,
        "power_model": power_model,
    }


def resolve_wake_model_inputs_for_flowfarm(flowfarm_model_options):
    """Resolve wake model options with defaults and validate user-provided values."""
    if flowfarm_model_options is None:
        flowfarm_model_options = {}
    if not isinstance(flowfarm_model_options, dict):
        raise TypeError("FLOWFarm options must be provided as a dictionary.")

    defaults = {
        "wake_deficit_model": "GaussYawVariableSpread",
        "wake_deflection_model": "GaussYawVariableSpreadDeflection",
        "wake_combination_model": "LinearLocalVelocitySuperposition",
        "local_turbulence_model": "LocalTIModelNoLocalTI",
        "tolerance": 1e-16,
    }

    allowed_values = {
        "wake_deficit_model": {
            "JensenTopHat",
            "JensenCosine",
            "MultiZone",
            "GaussOriginal",
            "GaussYaw",
            "GaussYawVariableSpread",
            "GaussSimple",
            "CumulativeCurl",
            "NoWakeDeficit",
        },
        "wake_deflection_model": {
            "NoYawDeflection",
            "GaussYawDeflection",
            "GaussYawVariableSpreadDeflection",
            "JiminezYawDeflection",
            "MultizoneDeflection",
        },
        "wake_combination_model": {
            "LinearFreestreamSuperposition",
            "SumOfSquaresFreestreamSuperposition",
            "SumOfSquaresLocalVelocitySuperposition",
            "LinearLocalVelocitySuperposition",
        },
        "local_turbulence_model": {
            "LocalTIModelNoLocalTI",
            "LocalTIModelMaxTI",
            "LocalTIModelGaussTI",
        },
    }

    unknown_keys = [k for k in flowfarm_model_options if k not in defaults]
    if unknown_keys:
        warnings.warn(
            f"FLOWFarm unknown wake model options {unknown_keys}; ignoring these keys.",
            UserWarning,
            stacklevel=2,
        )

    missing = [
        key
        for key in defaults
        if key not in flowfarm_model_options or flowfarm_model_options[key] is None
    ]
    if missing:
        defaults_used = {key: defaults[key] for key in missing}
        warnings.warn(
            f"FLOWFarm missing wake model inputs {missing}; using defaults {defaults_used}.",
            UserWarning,
            stacklevel=2,
        )

    resolved = {}
    model_keys = [
        "wake_deficit_model",
        "wake_deflection_model",
        "wake_combination_model",
        "local_turbulence_model",
    ]
    for key in model_keys:
        value = flowfarm_model_options.get(key, defaults[key])
        if not isinstance(value, str):
            raise TypeError(
                f"FLOWFarm option '{key}' must be a string. Got {type(value).__name__}."
            )

        value = value.strip()
        if not value:
            raise ValueError(f"FLOWFarm option '{key}' cannot be empty.")

        allowed_for_key = allowed_values[key]
        alias_lookup = {v.lower(): v for v in allowed_for_key}
        value_canonical = alias_lookup.get(value.lower())
        if value_canonical is None:
            raise ValueError(
                f"Invalid FLOWFarm option for '{key}': '{value}'. "
                f"Allowed values: {sorted(allowed_for_key)}"
            )

        resolved[key] = value_canonical

    tolerance = flowfarm_model_options.get("tolerance", defaults["tolerance"])
    if not isinstance(tolerance, (int, float)):
        raise TypeError(
            f"FLOWFarm option 'tolerance' must be numeric. Got {type(tolerance).__name__}."
        )
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("FLOWFarm option 'tolerance' must be > 0.")
    resolved["tolerance"] = tolerance

    return resolved


