# uav

Accurate UAV dynamics and control in Rust.

This package implements simple and accurate dynamical model for UAVs as well as a controller for such vehicle. Dynamics is done by integrating a system of differential equations for an UAV and contoller is a PID controller that integrates altitude, lateral, and body rate controllers.

Unfortunately physics engines (rapier, avian) couldn't simuilate dynamics of crazyflie, but this from-the-first principles approch works with best accuracy.