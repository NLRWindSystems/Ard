
## Installation instructions

<!-- `Ard` can be installed locally from the source code with `pip` or through a package manager from PyPI with `pip` or conda-forge with `conda`. -->
<!-- For Windows systems, `conda` is required due to constraints in the WISDEM installation system. -->
<!-- For macOS and Linux, any option is available. -->

`Ard` is currently in pre-release. It can be installed from PyPI or as a source-code installation.

### 1. Clone Ard source repository
If installing from PyPI, skip to [step 2.](#2.-Set-up-environment). If installing from source, the source can be cloned from github using the following command in your preferred location:
```shell
git clone git@github.com:NLRWindSystems/Ard.git
```
Once downloaded, you can enter the `Ard` root directory using
```shell
cd Ard
```

### 2. Set up environment
At this point, although not strictly required, we recommend creating a dedicated conda environment with `pip`, `python=3.12`, and `mamba` in it (except on apple silicon):

#### On Apple silicon
For Apple silicon, we recommend installing Ard natively.
```shell
CONDA_SUBDIR=osx-arm64 conda create -n ard-env 
conda activate ard-env
conda env config vars set CONDA_SUBDIR=osx-arm64 # this command makes the environment permanently native
conda install python=3.12
```

#### Or, on Intel
```shell
conda create --name ard-env
conda activate ard-env
conda install python=3.12 pip mamba -y
```

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
mamba install wisdem -y
pip install -e .[dev,docs]
```