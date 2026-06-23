import numpy as np

from ..floris import create_FLORIS_turbine_from_windIO
from .flowfarm_model import (
    ensure_flowfarm_loaded,
    resolve_turbine_inputs_for_flowfarm,
    resolve_wake_model_inputs_for_flowfarm,
    to_julia_vector_float64,
)

from .. import templates


class FLOWFarmComponent:

    def initialize(self):
        # This mixin is invoked explicitly by derived classes; no super() chain here.
        return

    def _get_air_density(self, wind_resource):
        return float(wind_resource.get("air_density", 1.225))

    def _get_wake_model_options(self, model_options):
        return resolve_wake_model_inputs_for_flowfarm(model_options.get("flowfarm", {}))

    def _build_wind_resource(
        self,
        jl,
        flowfarm_module,
        windrose_floris,
        ref_height,
        ref_air_density,
        wind_shear,
    ):
        if hasattr(windrose_floris, "wd_flat"):
            # WindRose (probability-based)
            dirs = np.deg2rad(np.asarray(windrose_floris.wd_flat))
            speeds = np.asarray(windrose_floris.ws_flat)
            probs = np.asarray(windrose_floris.freq_table_flat)
            n_states = len(speeds)
            mean_ti = float(np.mean(windrose_floris.ti_table_flat))
        else:
            # TimeSeries (temporal dispatch)
            dirs = np.deg2rad(np.asarray(windrose_floris.wind_directions))
            speeds = np.asarray(windrose_floris.wind_speeds)
            n_states = len(speeds)
            probs = np.full(n_states, 1.0 / n_states)
            mean_ti = float(np.mean(windrose_floris.turbulence_intensities))

        wind_dirs_rad = to_julia_vector_float64(jl, dirs)
        wind_speeds_vec = to_julia_vector_float64(jl, speeds)
        wind_probs_vec = to_julia_vector_float64(jl, probs)
        ambient_tis = jl.fill(mean_ti, n_states)
        measurementheight = jl.fill(float(ref_height), n_states)
        wind_shear_model = flowfarm_module.PowerLawWindShear(float(wind_shear))

        return flowfarm_module.DiscretizedWindResource(
            wind_dirs_rad,
            wind_speeds_vec,
            wind_probs_vec,
            measurementheight,
            float(ref_air_density),
            ambient_tis,
            wind_shear_model,
        )

    def _build_wake_model_set(self, flowfarm_module, wake_model_options):
        wake_deficit = getattr(
            flowfarm_module, wake_model_options["wake_deficit_model"]
        )()
        wake_deflection = getattr(
            flowfarm_module, wake_model_options["wake_deflection_model"]
        )()
        wake_combine = getattr(
            flowfarm_module, wake_model_options["wake_combination_model"]
        )()
        local_ti = getattr(
            flowfarm_module, wake_model_options["local_turbulence_model"]
        )()

        return flowfarm_module.WindFarmModelSet(
            wake_deficit,
            wake_deflection,
            wake_combine,
            local_ti,
        )

    def _create_update_fn(self, jl):
        jl.seval(
            """
            function ard_make_flowfarm_update_fn()
                return function (farm, x)
                    n = length(farm.turbine_x)
                    @inbounds for i in 1:n
                        farm.turbine_x[i] = x[i]
                        farm.turbine_y[i] = x[n + i]
                        farm.turbine_yaw[i] = x[2n + i]
                    end
                    return nothing
                end
            end
            """
        )
        return jl.ard_make_flowfarm_update_fn()

    def _build_farm_structures(
        self,
        jl,
        flowfarm_module,
        N_turbines,
        hub_height,
        rotor_diameter,
        generator_efficiency,
        cutin_wind_speed,
        cutout_wind_speed,
        rated_wind_speed,
        rated_power,
        windresource,
        ct_models,
        power_models,
        model_set,
        tolerance,
        x_init=None,
        y_init=None,
    ):
        # Use actual initial positions if provided so the sparsity pattern
        # computed by build_unstable_sparse_struct sees a non-degenerate farm
        # geometry (x0=zeros puts all turbines at the origin, hiding x/y deps).
        if x_init is None:
            x_init = np.zeros(N_turbines)
        if y_init is None:
            y_init = np.zeros(N_turbines)
        x0_np = np.concatenate([x_init, y_init, np.zeros(N_turbines)])
        x0 = to_julia_vector_float64(jl, x0_np)
        turbine_x = to_julia_vector_float64(jl, x_init)
        turbine_y = to_julia_vector_float64(jl, y_init)
        turbine_z = jl.zeros(N_turbines)
        turbine_yaw = jl.zeros(N_turbines)

        hub_heights = jl.fill(float(hub_height), N_turbines)
        rotor_diameters = jl.fill(float(rotor_diameter), N_turbines)
        generator_efficiencies = jl.fill(float(generator_efficiency), N_turbines)
        cut_in_speeds = jl.fill(float(cutin_wind_speed), N_turbines)
        cut_out_speeds = jl.fill(float(cutout_wind_speed), N_turbines)
        rated_speeds = jl.fill(float(rated_wind_speed), N_turbines)
        rated_powers = jl.fill(float(rated_power), N_turbines)
        update_fn = self._create_update_fn(jl)

        sparse_farm, sparse_struct = flowfarm_module.build_unstable_sparse_struct(
            x0,
            turbine_x,
            turbine_y,
            turbine_z,
            hub_heights,
            turbine_yaw,
            rotor_diameters,
            ct_models,
            generator_efficiencies,
            cut_in_speeds,
            cut_out_speeds,
            rated_speeds,
            rated_powers,
            windresource,
            power_models,
            model_set,
            update_fn,
            AEP_scale=1,
            opt_x=True,
            opt_y=True,
            opt_yaw=True,
            tolerance=tolerance,
        )

        farm = flowfarm_module.build_wind_farm_struct(
            x0,
            turbine_x,
            turbine_y,
            turbine_z,
            hub_heights,
            turbine_yaw,
            rotor_diameters,
            ct_models,
            generator_efficiencies,
            cut_in_speeds,
            cut_out_speeds,
            rated_speeds,
            rated_powers,
            windresource,
            power_models,
            model_set,
            update_fn,
            AEP_scale=1,
        )

        return x0, farm, sparse_farm, sparse_struct

    def _initial_turbine_positions(self, model_options, rotor_diameter):
        """Compute a non-degenerate initial grid layout for sparsity detection.

        build_unstable_sparse_struct computes the sparse Jacobian pattern by
        perturbing the design vector at the initial point. With x0=zeros (all
        turbines collocated at origin), x/y position changes create no wake
        effect, so those columns never appear in the pattern. We replicate the
        gridfarm formula with the configured spacing to get a spread-out layout.
        """
        layout = model_options.get("layout", {})
        spacing_prim = float(layout.get("spacing_primary", 7.0))
        spacing_sec = float(layout.get("spacing_secondary", spacing_prim))
        N = self.N_turbines
        N_sq = int(np.sqrt(N))

        cy, cx = np.meshgrid(
            np.arange(-((N_sq - 1) / 2), ((N_sq + 1) / 2)),
            np.arange(-((N_sq - 1) / 2), ((N_sq + 1) / 2)),
        )
        if N == N_sq ** 2:
            pass
        elif N <= N_sq * (N_sq + 1):
            # N is between N_sq² and N_sq*(N_sq+1): append a trailing row
            cx = np.vstack([cx, ((N_sq + 1) / 2) * np.ones((N_sq,))])
            cy = np.vstack([cy, np.arange(-((N_sq - 1) / 2), ((N_sq + 1) / 2))])
        else:
            # N is close to (N_sq+1)²: use a wider grid
            cy, cx = np.meshgrid(
                np.arange(-((N_sq) / 2), ((N_sq + 2) / 2)),
                np.arange(-((N_sq) / 2), ((N_sq + 2) / 2)),
            )

        x_init = (cx.ravel()[:N] * spacing_prim * rotor_diameter).astype(float)
        y_init = (cy.ravel()[:N] * spacing_sec * rotor_diameter).astype(float)
        return x_init, y_init

    def setup(self):
        jl = ensure_flowfarm_loaded()
        self._jl = jl
        model_options = self.options["modeling_options"]
        self.N_turbines = model_options["layout"]["N_turbines"]
        windIO = model_options["windIO_plant"]
        wind_resource = windIO["site"]["energy_resource"]["wind_resource"]

        turbine_floris = create_FLORIS_turbine_from_windIO(windIO)
        ref_air_density = self._get_air_density(wind_resource)

        hub_height = turbine_floris["hub_height"]
        rotor_diameter = turbine_floris["rotor_diameter"]

        windIOturbine = windIO["wind_farm"]["turbine"]
        turbine_inputs = resolve_turbine_inputs_for_flowfarm(windIOturbine)
        generator_efficiency = turbine_inputs["generator_efficiency"]
        rated_power = turbine_inputs["rated_power"]
        rated_wind_speed = turbine_inputs["rated_wind_speed"]
        cutin_wind_speed = turbine_inputs["cutin_wind_speed"]
        cutout_wind_speed = turbine_inputs["cutout_wind_speed"]
        ct_model = turbine_inputs["ct_model"]
        power_model = turbine_inputs["power_model"]

        wind_resource_dict = windIO["site"]["energy_resource"]["wind_resource"]
        resource_type = "timeseries" if "time" in wind_resource_dict else "probability"
        windrose_floris = templates.create_windresource_from_windIO(
            windIO,
            resource_type=resource_type,
        )

        ref_height = wind_resource.get("reference_height", hub_height)
        wind_shear = wind_resource.get("shear", 0.084)

        wake_model_options = self._get_wake_model_options(model_options)

        # FLOWFarm expects one model object per turbine.
        ct_models = jl.fill(ct_model, self.N_turbines)
        power_models = jl.fill(power_model, self.N_turbines)

        flowfarm_module = jl.FLOWFarm
        windresource = self._build_wind_resource(
            jl,
            flowfarm_module,
            windrose_floris,
            ref_height,
            ref_air_density,
            wind_shear,
        )
        model_set = self._build_wake_model_set(flowfarm_module, wake_model_options)

        x_init, y_init = self._initial_turbine_positions(model_options, rotor_diameter)
        x0, farm, sparse_farm, sparse_struct = self._build_farm_structures(
            jl,
            flowfarm_module,
            self.N_turbines,
            hub_height,
            rotor_diameter,
            generator_efficiency,
            cutin_wind_speed,
            cutout_wind_speed,
            rated_wind_speed,
            rated_power,
            windresource,
            ct_models,
            power_models,
            model_set,
            wake_model_options.get("tolerance", 1e-16),
            x_init=x_init,
            y_init=y_init,
        )

        self.flowfarm_module = flowfarm_module
        self.x0 = x0
        self.farm = farm
        self.sparse_farm = sparse_farm
        self.sparse_struct = sparse_struct

    def _build_design_vector(self, inputs):
        x_turbines = np.asarray(inputs["x_turbines"], dtype=float)
        y_turbines = np.asarray(inputs["y_turbines"], dtype=float)
        yaw_turbines = np.deg2rad(np.asarray(inputs["yaw_turbines"], dtype=float))
        return np.concatenate([x_turbines, y_turbines, yaw_turbines]).ravel()

    def _evaluate_sparse(self, x_eval_np):
        """Run sparse gradient evaluation once and cache AEP/gradient for reuse."""
        if hasattr(self, "_cached_sparse_x") and np.array_equal(
            self._cached_sparse_x, x_eval_np
        ):
            return

        jl = getattr(self, "_jl", None)
        if jl is None:
            jl = ensure_flowfarm_loaded()
            self._jl = jl
        x_eval = to_julia_vector_float64(jl, x_eval_np)
        calculate_grad_bang = getattr(self.flowfarm_module, "calculate_aep_gradient!")
        aep_val, grad_val = calculate_grad_bang(
            self.sparse_farm,
            x_eval,
            self.sparse_struct,
        )

        self._cached_sparse_x = x_eval_np.copy()
        self._cached_sparse_aep = float(np.asarray(aep_val).ravel()[0])
        self._cached_sparse_grad = np.asarray(grad_val).ravel().copy()

    def _evaluate_farm(self, x_eval_np):
        """Run regular farm AEP evaluation and cache AEP."""
        if hasattr(self, "_cached_farm_x") and np.array_equal(
            self._cached_farm_x, x_eval_np
        ):
            return

        jl = getattr(self, "_jl", None)
        if jl is None:
            jl = ensure_flowfarm_loaded()
            self._jl = jl
        x_eval = to_julia_vector_float64(jl, x_eval_np)
        calculate_aep_bang = getattr(self.flowfarm_module, "calculate_aep!")
        aep_val = calculate_aep_bang(self.farm, x_eval)

        self._cached_farm_x = x_eval_np.copy()
        self._cached_farm_aep = float(np.asarray(aep_val).ravel()[0])

    def _compute_aep(self, inputs, outputs):
        """Compute farm AEP using regular calculate_aep!(farm, x)."""
        x_eval_np = self._build_design_vector(inputs)
        self._evaluate_farm(x_eval_np)
        outputs["AEP_farm"] = self._cached_farm_aep

    def _compute_aep_partials(self, inputs, partials):
        """Compute AEP partial derivatives from sparse gradient evaluation."""
        x_eval_np = self._build_design_vector(inputs)
        self._evaluate_sparse(x_eval_np)
        grad = self._cached_sparse_grad
        partials["AEP_farm", "x_turbines"] = grad[: self.N_turbines]
        partials["AEP_farm", "y_turbines"] = grad[self.N_turbines : 2 * self.N_turbines]
        partials["AEP_farm", "yaw_turbines"] = grad[
            2 * self.N_turbines : 3 * self.N_turbines
        ]


class FLOWFarmAEP(templates.FarmAEPTemplate, FLOWFarmComponent):

    def initialize(self):
        templates.FarmAEPTemplate.initialize(self)
        FLOWFarmComponent.initialize(self)

    def setup(self):
        templates.FarmAEPTemplate.setup(self)
        FLOWFarmComponent.setup(self)

    def setup_partials(self):
        self.declare_partials("AEP_farm", "x_turbines", method="exact")
        self.declare_partials("AEP_farm", "y_turbines", method="exact")
        self.declare_partials("AEP_farm", "yaw_turbines", method="exact")

    def compute(self, inputs, outputs):
        FLOWFarmComponent._compute_aep(self, inputs, outputs)

    def compute_partials(self, inputs, partials):
        FLOWFarmComponent._compute_aep_partials(self, inputs, partials)


class FLOWFarmBatchPower(templates.BatchFarmPowerTemplate, FLOWFarmComponent):

    def initialize(self):
        templates.BatchFarmPowerTemplate.initialize(self)
        FLOWFarmComponent.initialize(self)

    def setup(self):
        templates.BatchFarmPowerTemplate.setup(self)
        FLOWFarmComponent.setup(self)

    def setup_partials(self):
        # State power sensitivities are provided via sparse_struct.state_gradients.
        self.declare_partials("power_farm", "x_turbines", method="exact")
        self.declare_partials("power_farm", "y_turbines", method="exact")
        self.declare_partials("power_farm", "yaw_turbines", method="exact")

    def compute(self, inputs, outputs):
        x_eval_np = self._build_design_vector(inputs)
        self._evaluate_sparse(x_eval_np)

        state_powers = np.asarray(self.sparse_struct.state_powers).ravel()
        turbine_powers = np.asarray(self.sparse_struct.turbine_powers)

        outputs["power_farm"] = state_powers
        outputs["AEP_farm"] = float(state_powers.sum()) * 3600.0  # W*h, assuming 1-hour timesteps
        if (
            self.options["modeling_options"]
            .get("aero", {})
            .get("return_turbine_output")
        ):
            outputs["power_turbines"] = turbine_powers
            outputs["thrust_turbines"] = np.zeros(
                (self.N_turbines, self.N_wind_conditions)
            )

    def compute_partials(self, inputs, partials):
        x_eval_np = self._build_design_vector(inputs)
        self._evaluate_sparse(x_eval_np)

        state_gradients = np.asarray(self.sparse_struct.state_gradients)
        partials["power_farm", "x_turbines"] = state_gradients[:, : self.N_turbines]
        partials["power_farm", "y_turbines"] = state_gradients[
            :, self.N_turbines : 2 * self.N_turbines
        ]
        partials["power_farm", "yaw_turbines"] = state_gradients[
            :, 2 * self.N_turbines : 3 * self.N_turbines
        ]
