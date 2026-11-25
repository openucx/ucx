# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import ucxx


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(ucxx.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(ucxx.__version__, str)
    assert len(ucxx.__version__) > 0
