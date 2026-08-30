"""Shared helpers for the OpenCL backend tests. Not a test module.

Importing this module requires pyopencl, so every test module must guard
with ``pytest.importorskip("pyopencl")`` (and a platform-availability check)
*before* importing from here.

The shipped OpenCL backend contains device functions only; the ``__kernel``
wrappers needed to test them are defined as string literals in the test
modules and appended to the packaged source by :func:`build`.
"""

import numpy as np
import pyopencl as cl

from meepmeep.backends.opencl import read_kernel_source, build_options

_CONTEXT = None
_QUEUE = None


def get_context() -> cl.Context:
    """Return a lazily-created OpenCL context shared by the test session."""
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = cl.create_some_context(interactive=False)
    return _CONTEXT


def get_queue() -> cl.CommandQueue:
    """Return the command queue for the shared test context."""
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = cl.CommandQueue(get_context())
    return _QUEUE


def has_fp64(ctx: cl.Context | None = None) -> bool:
    """True if every device in the context supports cl_khr_fp64."""
    ctx = ctx if ctx is not None else get_context()
    return all(device.double_fp_config for device in ctx.devices)


def build(test_kernels: str, *source_files: str, precision: str = 'double') -> cl.Program:
    """Build the named packaged sources plus test-only kernel wrappers."""
    source = read_kernel_source(*source_files) + '\n' + test_kernels
    return cl.Program(get_context(), source).build(options=build_options(precision))


def real_dtype(precision: str):
    """NumPy dtype matching a REAL build option."""
    return np.float64 if precision == 'double' else np.float32


def upload(queue: cl.CommandQueue, array, precision: str = 'double') -> cl.Buffer:
    """Upload a float array as a flattened read-only REAL device buffer."""
    host = np.ascontiguousarray(array, dtype=real_dtype(precision)).ravel()
    return cl.Buffer(queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host)


def upload_ep_table(queue: cl.CommandQueue, ep_table) -> cl.Buffer:
    """Upload an expansion-point lookup table as the int32 buffer the device expects."""
    host = np.ascontiguousarray(ep_table, dtype=np.int32)
    return cl.Buffer(queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host)


def output_buffer(queue: cl.CommandQueue, n: int, precision: str = 'double') -> tuple[cl.Buffer, np.ndarray]:
    """Allocate an n-element REAL output buffer and its host read-back array."""
    host = np.empty(n, dtype=real_dtype(precision))
    return cl.Buffer(queue.context, cl.mem_flags.WRITE_ONLY, host.nbytes), host


def read_back(queue: cl.CommandQueue, buffer: cl.Buffer, host: np.ndarray) -> np.ndarray:
    """Blocking read of a device buffer into its host array."""
    cl.enqueue_copy(queue, host, buffer)
    queue.finish()
    return host
