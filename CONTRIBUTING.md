# Contributing to MeepMeep

Thanks for your interest in MeepMeep! Bug reports, questions, documentation
fixes and code are all welcome. This guide covers how to report problems,
set up a development checkout, and get a pull request merged.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and asking questions

Open an issue on [GitHub](https://github.com/hpparvi/meepmeep/issues). A
good bug report includes:

- the MeepMeep version (`python -c "import meepmeep; print(meepmeep.__version__)"`),
  plus your Python, NumPy and Numba versions (and JAX or PyOpenCL if relevant);
- a minimal, self-contained example that reproduces the problem;
- what you expected and what you got instead.

For numerical problems, include the orbital parameters (`tc`/`tp`, `p`, `a`,
`i`, `e`, `w`, `lan`), the number of expansion points, and the times you
evaluated. Accuracy depends strongly on eccentricity and on the orbital
phase, so these details matter.

## Development setup

```bash
git clone https://github.com/hpparvi/meepmeep.git
cd meepmeep
pip install -e ".[test]"
```

Add the extras you need: `jax` for the JAX backend, `opencl` for the OpenCL
backend, and `docs` for building the documentation, e.g.
`pip install -e ".[test,jax,docs]"`.

## Running the tests

```bash
pytest meepmeep/tests/                 # everything
pytest meepmeep/tests/ -m "not slow"   # skip the long-running tests
```

The optional suites skip themselves when their dependency is missing: the
JAX tests without `jax`, the OpenCL tests without `pyopencl` and an OpenCL
platform, and the C library tests without a C compiler on `PATH`.

Measure coverage with the JIT disabled, since compiled Numba kernels are
invisible to the tracer:

```bash
NUMBA_DISABLE_JIT=1 pytest -m "not slow" --cov
```

## Making changes

### API stability

The stable public API is:

- the high-level classes (`Orbit`, `Expansion2D`, `Expansion3D`);
- the low-level aggregators `meepmeep.numba2d` and `meepmeep.numba3d`
  (their `__all__`);
- the public names (no leading underscore) of the modules under
  `meepmeep.backends.numba` and `meepmeep.backends.opencl`.

Breaking changes to these are possible when they clearly improve clarity or
usability, but they must be recorded under a **Breaking** note in
[`CHANGELOG.md`](CHANGELOG.md). Names with a leading underscore are private.
The JAX backend (`meepmeep.jax2d`, `meepmeep.jax3d`, `meepmeep.backends.jax`)
is experimental and may still change.

### Keeping the backends in step

MeepMeep implements the same quantities in several places: the Numba
backend (the reference), the OpenCL device functions (which also compile as
the C library in `c/`), and the JAX backend. A change to a Numba kernel's
values usually needs porting to the `.cl` sources and to the JAX module as
well. After changing a signature in a `.cl` file, regenerate the C header
with `python c/tools/generate_header.py`.

The naming scheme is documented in
[`docs/source/naming_conventions.rst`](docs/source/naming_conventions.rst).
[`CLAUDE.md`](CLAUDE.md) doubles as a detailed developer guide, including a
step-by-step checklist for adding a new Taylor-series quantity.

### Testing numerical code

- Compare new evaluators against the exact Newton-Raphson references, and
  gradients against the existing `_d`/`_od` kernels or finite differences.
- Use time grids that span several epochs and have at least 16 samples.
  Single-epoch or few-sample grids can miss errors in the period-folding
  terms.
- Mutation-test new code: flip a sign or perturb a coefficient, and check
  that a test fails.

### Code and documentation style

- Docstrings follow the NumPy style, with units in square brackets at the
  end of each description (`[days]`, `[radians]`, `[R_star]`).
- Call the sky-projected star-planet distance the "projected separation".
- Keep docstrings and documentation in ASCII, and avoid em-dashes.
- Evaluate polynomials with Horner's method.

Build the documentation with `cd docs && make html`; the output goes to
`docs/build/html/`.

## Pull requests

- Keep each pull request focused on one change, and describe what it does
  and why.
- Add or update tests for any behaviour you change, and make sure
  `pytest meepmeep/tests/ -m "not slow"` passes.
- Add an entry under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) for
  user-visible changes.
- Update the documentation when you change public behaviour. The narrative
  docs embed function names and examples that are not checked by the build.

## Generative AI policy

Contributions developed with LLMs are welcome, but each one needs a human
author who oversees the work and takes responsibility for it. Purely
agentic contributions, made without human oversight, are not accepted. See
the [Generative AI policy](README.md#generative-ai-policy) in the README.

## License

MeepMeep is released under the GNU General Public License v3.0. By
contributing, you agree that your contributions are licensed under the same
terms.
