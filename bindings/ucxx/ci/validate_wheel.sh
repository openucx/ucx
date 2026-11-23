#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

cd "${package_dir}"

# Make glob expand to nothing if there are no matches
shopt -s nullglob

# Pure wheels can be installed on any OS and we want to avoid users being able
# to begin installing them on Windows or OSX when we know that the dependencies
# won't work / be available.
for wheel in "${wheel_dir_relative_path}"/*-py3-none-any.whl; do
    rapids-logger "Retagging pure Python wheel: ${wheel}"

    # Pull the version of GLIBC used in the wheel build container
    glibc_version=$(python -c 'import os; print(os.confstr("CS_GNU_LIBC_VERSION").split()[-1].replace(".", "_"))')
    wheel_tag_template=manylinux_LIBCVER_x86_64.manylinux_LIBCVER_aarch64

    # Retag for manylinux x86_64 and manylinux aarch64
    # substituting in the glibc_version gathered above
    wheel tags --platform-tag=${wheel_tag_template//LIBCVER/$glibc_version} --remove "${wheel}"

    rapids-logger "Successfully retagged wheel for manylinux platforms"
done

rapids-logger "validate packages with 'pydistcheck'"

pydistcheck \
    --inspect \
    "${wheel_dir_relative_path}"/*.whl

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "${wheel_dir_relative_path}"/*.whl

rapids-logger "validate packages with 'abi3audit'"

# 'abi3audit' fails on wheels that contain DSOs but don't have an ABI tag (like 'libucxx').
# This '-name' filtering avoids trying those.
find \
    "${wheel_dir_relative_path}" \
    -type f \
    -name '*abi*' \
    -exec abi3audit --strict --summary --verbose '{}' \+
