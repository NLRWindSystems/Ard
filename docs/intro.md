
# Introduction

**Dig in to wind farm design.**

<!-- The (aspirationally) foolproof tool for preparing wind farm layouts. -->

[An ard is a type of simple and lightweight plow](https://en.wikipedia.org/wiki/Ard_\(plough\)), used through the single-digit centuries to prepare a farm for planting.
The intent of `Ard` is to be a modular, full-stack multi-disciplinary optimization tool for wind farms.

Wind farms are complicated, multi-disciplinary systems.
They are aerodynamic machines (composed of complicated control systems, power electronic devices, etc.), social and political objects, generators of electrical power and consumers of electrical demand, and the core value generator (and cost) of complicated financial instruments.
Moreover, the design of any *one* of these aspects affects all the rest!

`Ard` is a platform for wind farm layout optimization that seeks to enable plant-level design choices that can incorporate these different aspects _and their interactions_ to make wind energy projects more successful.
In brief, we are designing `Ard` to be: principled, modular, extensible, and effective, to allow resource-specific wind farm layout optimization with realistic, well-posed constraints, holistic and complex objectives, and natural incorporation of multiple fidelities and disciplines.

## Documentation
The documentation is organized as follows:
- **Getting Started**: helpful information to get you started using `Ard`, including installation instructions
<!-- - **Tutorials**: example problems with solutions
- **How-to Guides**: instructions on solving particular problems and using specific features of `Ard` -->
- **Reference**: dry, but hopefully useful, reference material including the API reference manual
- **Explanation**: discursive content on the whys, hows, and the wherefores of `Ard` for those that are interested

## Design philosophy

The design of `Ard` was inspired by two use cases in particular:
1) systems energy researchers who are focusing on one specific subdiscipline (e.g. layout strategies, social impacts, or aerodynamic modeling) but want to be able to easily keep track of how a change in one discipline impacts the entire value chain down to production, cost, value, and/or societal outcomes of energy or even optimize with respect to these, and
2) private industry researchers who run business cases and may want to drop in proprietary analysis modules for specific disciplines while preserving some of the open-source modules of `Ard`.

`Ard` is being developed as a modular tool to enable these types of research queries.
The goals during the development of `Ard` are to be:
1) principled:
   - robustly documented
   - adhering to [best-practices for code development](https://doi.org/10.2172/2479115)
2) modular and extensible:
   - choose the analysis components you want
   - skip the ones you don't
   - build yourself the ones we don't have
3) effective
    - robustly tested and testable at both unit and system levels
These principles guide us to implement, using [`OpenMDAO`](https://openmdao.org) as a backbone, a multi-disciplinary design, analysis, and optimization (MDAO) model of the wind farm layout problem, a toolset to accomplish the capability goals of `Ard`, to:
1) allow optimization of wind farm layouts for specific wind resource profiles
2) enable the incorporation of realistic but well-posed constraints
3) target holistic and complex system-level optimization objectives like LCOE and beyond-LCOE metrics
4) naturally incorporate analyses across fidelities to efficiently integrate advanced simulation

## Current capabilities

For the beta pre-release of `Ard`, we concentrate on optimization problems for wind plants, starting from structured layouts to minimize LCOE.
This capability is demonstrated for a land-based (LB) wind farm in `examples/01_onshore` and tested in an abridged form in `test/system/ard/api/test_LCOE_LB_stack.py`.
In this example, the wind farm layout is parametrized with two angles, named orientation and skew, and turbine distancing for rows and columns.
Additionally, we have offshore examples adjacent to the onshore example in the `examples` subdirectory.
In the beta pre-release stage, the constituent subcomponents of these problems are known to work and have full testing coverage.

These cases start from a four parameter farm layout, compute land use area, make FLORIS estimates of annual energy production (AEP), compute turbine capital costs, balance-of-station (BOS), and operational costs elements of NREL's turbine systems engineering tool [WISDEM](https://github.com/wisdem/wisdem), and finally give summary estimates of plant finance figures.
The components that achieve this can be assembled to either run a single top-down analysis run, or run an optimization.

---

Copyright &copy; 2024, Alliance for Sustainable Energy, LLC
