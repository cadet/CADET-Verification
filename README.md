# CADET-Verification

This repository is subjected to the CADET-Core simulator and includes a comprehensive suite of tests that extend beyond the scope of a typical CI pipeline.
CADET-Verification verifies the implementation, models and methods within CADET-Core via order-of-convergence tests and the recreation of validated case studies.
CADET-Verification is part of the deployment pipeline of CADET-Core and must additionally be run on demand, i.e. when critical changes are made to the simulator.

This repository is solely dedicated to verification.
Computational performance benchmarks are defined and tracked in the [CADET-Benchmark](https://github.com/cadet/CADET-Benchmark) repository.

The research data generated by CADET-Verification is automatically managed using [CADET-RDM](https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/CADET-RDM).
The results of the verification studies can be accessed in the [CADET-Verification-Output](https://github.com/cadet/CADET-Verification-Output) repository.


## Usage

CADET-Verification must be run for deployment and on demand, specifically when critical changes are made to the simulator.

The tests are located in the `src` folder and have descriptive names, so that running a subset of the tests is also possible.

The reference data used in some of the tests implemented in CADET-Core can be generated using the corresponding tests defined in CADET-Verification.

## Contributing

We welcome and appreciate all contributions!

If you are a CADET-Core developer adding a new feature, you must also include an appropriate set of verification tests or case studies in this repository.
This ensures that your contribution meets the quality standards and helps maintain the long-term reliability and maintainability of the project.

Furthermore, contributions aimed at improving or extending the evaluation functions are highly encouraged.