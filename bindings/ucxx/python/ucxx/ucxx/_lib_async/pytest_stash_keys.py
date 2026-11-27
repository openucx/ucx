# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest stash keys shared between ``conftest.py`` and ``utils_test``."""

import pytest

# In-process storage for the effective asyncio timeout (see
# ``_lib_async/tests/conftest.py``). ``pytest.Config.cache`` is unreliable on
# ``pytest-xdist`` workers; ``config.stash`` is not.
ASYNCIO_PLUGIN_TIMEOUT_STASH_KEY = pytest.StashKey[float]()
