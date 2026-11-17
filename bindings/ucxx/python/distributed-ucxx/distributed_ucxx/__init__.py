# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from .ucxx import (
    UCXXBackend,
    UCXXBackendLegacyPrefix,
)


from ._version import __git_commit__, __version__

__all__ = [
    "UCXXBackend",
    "UCXXBackendLegacyPrefix",
    "__git_commit__",
    "__version__",
]
