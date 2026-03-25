# COMP0217 Sensor Fusion

Notes on running/using the code

- There are very slight changes between the simulink embedded function and the `myEKF_ca.m` file to do with initialisation, as the `.m` file was used in certain tests (so has configuration variables) and has a seeded orientation

- The simulink model has been slightly adapted, making the selector block map 9->2 rather than 8->2 due to our state being a 9x1 vector. Additionally, the function block was fixed to an output of size 9 to help with the code generation stage though it *should* be able to infer it as normal

- The `run_simulink` file provides an interface to load and run the simulink file, though it does not compensate for incorrect initial rotation - the default is -π\2, but the MSE benefits from any and all correction in this initial guess, not just a post-rotation onthe path and heading. 