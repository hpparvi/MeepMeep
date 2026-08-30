---
name: meepmeep
description: Use when writing or debugging code that computes exoplanet orbits with the MeepMeep package (Orbit, Expansion2D/3D, numba2d/numba3d primitives, or the OpenCL device-function backend) - transit geometry, projected separations, radial velocities, phase curves, and their analytic gradients.
---

Read `reference.md` in this skill directory before writing MeepMeep-consuming
code. It is the consumer-facing API cheatsheet: the stability contract and
sanctioned imports, unit and coordinate conventions, gradient layout and
timing-basis rules, the high- and low-level APIs, the OpenCL backend's
device-function conventions (including the numba-twin comment convention),
and the pitfalls agents most often get wrong.

The file is a synced snapshot of `docs/llms.md` in the MeepMeep repository;
if the two disagree, the repository copy wins.
