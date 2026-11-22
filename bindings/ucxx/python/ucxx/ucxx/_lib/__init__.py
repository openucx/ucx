# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from .libucxx import _create_exceptions

# Ensure Python exceptions are created before use
_create_exceptions()
