# Unit and system testing

The installation can be tested comprehensively using `pytest` from the top-level directory.
The developers also provide some convenience scripts for testing new installations; from the `Ard` folder run unit and regression tests:
```shell
source test/run_local_test_unit.sh
source test/run_local_test_system.sh
```
These enable the generation of HTML-based coverage reports by default and can be used to track "coverage", or the percentage of software lines of code that are run by the testing systems.
`Ard`'s git repository includes requirements for both the `main` and `develop` branches to have 80% coverage on unit testing and 50% testing in system testing, which are, respectively, tests of individual parts of `Ard` and "systems" composed of multiple parts.
Failures are not tolerated in code that is merged onto these branches and code found therein *should* never cause a testing failure if it has been found there.
If the process of installation and testing fails, please open a new issue [here](https://github.com/NLRWindSystems/Ard/issues).
If the process of installation and testing fails, please open a new issue [here](https://github.com/NLRWindSystems/Ard/issues).
