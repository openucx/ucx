# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""UCXX: Python bindings for the Unified Communication X library
(UCX <www.openucx.org>)"""

import logging
import os

# If libucx was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libucxx
except ModuleNotFoundError:
    pass
else:
    libucxx.load_library()
    del libucxx


logger = logging.getLogger("ucx")

# Notice, if we have to update environment variables we need to do it
# before importing UCX, which must happen also before the Cython code
# import to prevent UCS unused variable warnings.
if "UCX_MEMTYPE_CACHE" not in os.environ:
    # See <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>
    logger.debug("Setting env UCX_MEMTYPE_CACHE=n, which is required by UCX")
    os.environ["UCX_MEMTYPE_CACHE"] = "n"

from . import exceptions, types, testing  # noqa: E402
from ._lib import libucxx  # noqa: E402
from .core import *  # noqa: E402, F403
from .utils import get_address, get_ucxpy_logger  # noqa: E402

try:
    import pynvml
except ImportError:
    pynvml = None

_ucx_version = get_ucx_version()  # noqa: F405
__ucx_min_version__ = "1.20.0"
__ucx_version__ = "%d.%d.%d" % _ucx_version

if _ucx_version < tuple(int(i) for i in __ucx_min_version__.split(".")):
    raise ImportError(
        f"Support for UCX {__ucx_version__} has ended. Please upgrade to "
        f"{__ucx_min_version__} or newer. If you believe the wrong version "
        "is being loaded, please check the path from where UCX is loaded "
        "by rerunning with the environment variable `UCX_LOG_LEVEL=debug`."
    )

# Setup UCX-Py logger
logger = get_ucxpy_logger()

if "UCX_RNDV_THRESH" not in os.environ:
    logger.info("Setting UCX_RNDV_THRESH=8192")
    os.environ["UCX_RNDV_THRESH"] = "8192"

if "UCX_RNDV_FRAG_MEM_TYPE" not in os.environ:
    logger.info("Setting UCX_RNDV_FRAG_MEM_TYPE=cuda")
    os.environ["UCX_RNDV_FRAG_MEM_TYPE"] = "cuda"

if (
    pynvml is not None
    and "UCX_CUDA_COPY_MAX_REG_RATIO" not in os.environ
    and _ucx_version >= (1, 12, 0)
):
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        large_bar1 = [False] * device_count

        def _is_mig_device(handle):
            try:
                pynvml.nvmlDeviceGetMigMode(handle)[0]
            except pynvml.NVMLError:
                return False
            return True

        for dev_idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)

            try:
                total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            except pynvml.NVMLError_NotSupported:
                total_memory = None

            # Ignore MIG devices and devices with no memory resource (i.e., only
            # integrated CPU+GPU memory resource) and rely on UCX's default for
            # now. Increasing `UCX_CUDA_COPY_MAX_REG_RATIO` should be thoroughly
            # tested, as it's not yet clear whether it would be safe to set `1.0`
            # for those instances too.
            if _is_mig_device(handle) or total_memory is None:
                continue

            try:
                bar1_total = pynvml.nvmlDeviceGetBAR1MemoryInfo(handle).bar1Total
            except pynvml.NVMLError_NotSupported:
                # Bar1 access not supported on this device, set it to
                # zero (always lower than device memory).
                bar1_total = 0

            if total_memory <= bar1_total:
                large_bar1[dev_idx] = True

        if all(large_bar1):
            logger.info("Setting UCX_CUDA_COPY_MAX_REG_RATIO=1.0")
            os.environ["UCX_CUDA_COPY_MAX_REG_RATIO"] = "1.0"
    except (
        pynvml.NVMLError_LibraryNotFound,
        pynvml.NVMLError_DriverNotLoaded,
        pynvml.NVMLError_Unknown,
    ):
        pass

if "UCX_MAX_RNDV_RAILS" not in os.environ and _ucx_version >= (1, 12, 0):
    logger.info("Setting UCX_MAX_RNDV_RAILS=1")
    os.environ["UCX_MAX_RNDV_RAILS"] = "1"

if "UCX_PROTO_ENABLE" not in os.environ and (1, 12, 0) <= _ucx_version < (1, 18, 0):
    # UCX protov2 still doesn't support CUDA async/managed memory
    logger.info("Setting UCX_PROTO_ENABLE=n")
    os.environ["UCX_PROTO_ENABLE"] = "n"


from ._version import __git_commit__, __version__  # noqa: E402

__all__ = [
    "exceptions",
    "types",
    "testing",
    "libucxx",
    *core.__all__,  # noqa: F405
    "__git_commit__",
    "__version__",
    "get_address",
]
