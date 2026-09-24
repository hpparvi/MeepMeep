# libmeepmeep

The MeepMeep Taylor-series orbit evaluators as a plain C99 library. This
directory holds the CMake build, the public header, the C-only sources
(expansion-point placement, orbit-wide solvers, gradient basis transforms)
and an example; the evaluators themselves are compiled from the sources
shared with the OpenCL backend in `../meepmeep/backends/opencl/`.

```bash
cmake -S c -B c/build -DMEEPMEEP_BUILD_EXAMPLES=ON
cmake --build c/build
./c/build/transit
cmake --install c/build --prefix "$HOME/.local"
```

Then `#include <meepmeep.h>` and link with `-lmeepmeep -lm`. Precision is
fixed to double. Do not build with `-ffast-math`.

- `include/meepmeep.h`: the public API. The block between the
  `GENERATED PROTOTYPES` markers is produced by `tools/generate_header.py`
  from the `.cl` sources; rerun it after changing a shared signature
  (`meepmeep/tests/test_c_library.py` fails while it is stale).
- `src/meepmeep.c`: unity build of the shared sources.
- `src/expansion_points.c`, `src/orbit.c`: the C-only functions.
- `examples/transit.c`: the full-orbit pipeline end to end.

Full documentation: `docs/source/c_library.rst`.
