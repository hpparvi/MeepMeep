/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Unity build of the sources shared with the OpenCL backend.
 *
 *  Every function below is defined in meepmeep/backends/opencl (the .cl files), where
 *  the MM_INLINE / MM_GLOBAL / REAL macros (set up at the top of common.cl
 *  when __OPENCL_VERSION__ is absent) turn the device functions into
 *  ordinary external C99 functions on double. Including the public header
 *  first makes the compiler check every definition against its prototype,
 *  so a signature change in a .cl file that has not been propagated to the
 *  header is a compile error rather than silent drift.
 *
 *  solve_kernels.cl is OpenCL-only (__kernel entry points) and is not
 *  included. The order below is the dependency order of
 *  `meepmeep.backends.opencl.SOURCE_FILES`.
 */

#define REAL double
#include "meepmeep.h"

#include "common.cl"
#include "solve2d.cl"
#include "solve3d.cl"
#include "point2d.cl"
#include "point2dd.cl"
#include "point3d.cl"
#include "point3dd.cl"
#include "orbit3d.cl"
#include "orbit3dd.cl"
