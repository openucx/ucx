# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import distributed_ucxx


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(distributed_ucxx.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(distributed_ucxx.__version__, str)
    assert len(distributed_ucxx.__version__) > 0
