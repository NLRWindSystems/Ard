
# Ard

[![CI/CD test suite](https://github.com/WISDEM/Ard/actions/workflows/python-tests-consolidated.yaml/badge.svg?branch=main)](https://github.com/WISDEM/Ard/actions/workflows/python-tests-consolidated.yaml)

![Ard logo](assets/logomaker/logo.png)

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

Ard documentation is available at [https://wisdem.github.io/Ard](https://wisdem.github.io/Ard)

## Installation instructions

<!-- `Ard` can be installed locally from the source code with `pip` or through a package manager from PyPI with `pip` or conda-forge with `conda`. -->
<!-- For Windows systems, `conda` is required due to constraints in the WISDEM installation system. -->
<!-- For macOS and Linux, any option is available. -->

`Ard` is currently in pre-release. It can be installed from PyPI or as a source-code installation.

### 1. Clone Ard source repository

If installing from PyPI, skip to [step 2.](#2.-Set-up-environment). If installing from source, the source can be cloned from github using the following command in your preferred location:
```shell
git clone git@github.com:WISDEM/Ard.git
```
Once downloaded, you can enter the `Ard` root directory using
```shell
cd Ard
```

### 2. Set up environment

At this point, although not strictly required, we recommend creating a dedicated conda environment with `pip` and `python=3.12` in it:

```shell
conda create --name ard-env
conda activate ard-env
conda install python=3.12 pip -y
```

#### On Apple silicon

For Apple silicon, the above will install for the default ("native") architecture.

For some advanced users, installation specifying non-default architecture may be desirable for compatibility:
```shell
# ARCH_SPEC=osx-arm64  # native AMD on apple silicon, if Intel x64 is defaulted
ARCH_SPEC=osx-64  # Intel x64 for Apple Binary Interface
CONDA_SUBDIR=$ARCH_SPEC conda create --name ard-env
conda activate ard-env
conda env config vars set CONDA_SUBDIR=$ARCH_SPEC # this command makes the environment permanently native
conda install python=3.12 pip -y
```
This can enable interfacing compatibly to software that has not been built for Apple silicon yet.

### 3. Install Ard

From here, installation can be handled by `pip`.

#### To install from PyPI

```shell
pip install ard-nrel
```

#### For a basic and static installation from source, run:

```shell
pip install .
```

#### For development (and really for everyone during pre-release), we recommend a full development installation from source:

```shell
pip install -e .[dev,docs]
```
which will install in "editable mode" (`-e`), such that changes made to the source will not require re-installation, and with additional optional packages for development and documentation (`[dev,docs]`).

#### If you have problems with WISDEM not installing correctly

There can be some hardware-software mis-specification issues with WISDEM installation from `pip` for MacOS 12 and 13 on machines with Apple Silicon.
In the event of issues, WISDEM can be installed manually or using `conda` without issues, then `pip` installation can proceed.

```shell
conda install wisdem -y
pip install -e .[dev,docs]
```

## Testing instructions

The installation can be tested comprehensively using `pytest` from the top-level directory.
The developers also provide some convenience scripts for testing new installations; from the `Ard` folder run unit and regression tests:
```shell
source test/run_local_test_unit.sh
source test/run_local_test_system.sh
```
These enable the generation of HTML-based coverage reports by default and can be used to track "coverage", or the percentage of software lines of code that are run by the testing systems.
`Ard`'s git repository includes requirements for both the `main` and `develop` branches to have 80% coverage on unit testing and 50% testing in system testing, which are, respectively, tests of individual parts of `Ard` and "systems" composed of multiple parts.
Failures are not tolerated in code that is merged onto these branches and code found therein *should* never cause a testing failure if it has been found there.
If the process of installation and testing fails, please open a new issue [here](https://github.com/WISDEM/Ard/issues).

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

# Contributing to `Ard`

We have striven towards best-practices documentation and testing for `Ard`.
Contribution is welcome, and we are happy [to field pull requests from github](https://github.com/WISDEM/Ard/pulls).
For acceptance, PRs must:
- be formatted using [`black`](https://github.com/psf/black)
- not fail any unit tests or system tests
- achieve coverage criteria for unit & system testing
- be documented enough for continued maintenance by core `Ard` developers

## Building Documentation

To build the documentation locally, run the following from the top-level `Ard/` directory:
```shell
jupyter-book build docs/
```
You can then open `Ard/docs/_build/html/index.html` to view the docs.

---

Released as open-source software by the National Renewable Energy Laboratory under NREL software record number SWR-25-18.

Copyright &copy; 2024, Alliance for Sustainable Energy, LLC.
