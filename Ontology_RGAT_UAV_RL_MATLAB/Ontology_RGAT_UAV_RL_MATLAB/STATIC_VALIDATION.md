# Static validation report

- MATLAB package symbols discovered: `54`.
- MATLAB source files: `61`.
- Cross-package unresolved references: `0`.
- Default hover rotor speed arithmetic check: `448.523 rad/s`.
- Default max rotor speed from 30 N total thrust: `645.497 rad/s`.
- `4*kT*omega_hover^2 = 14.484422 N`, matching `m*g = 14.484422 N`.
- MATLAB executable available in this build environment: `no`.
- Octave executable available in this build environment: `no`.

Because no MATLAB/Octave runtime is present here, neural training and MATLAB graphics were **not executed in this environment**. Runtime physics assertions are implemented in `validation/validatePhysics.m` and should be run first with `run_quick_smoke_test.m`.
