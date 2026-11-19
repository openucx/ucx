#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#
# Set the wire-compat test matrix as the "matrix" output variable: the latest
# <num_releases> vX.Y.x release branches older than the version in configure.ac.
#
# Usage: wire_compat_matrix.sh <num_releases>
# Env:   BUILD_REPOSITORY_URI - repository to take the release branches from
#

num_releases=${1:?Usage: $0 <num_releases>}

major=$(sed -n 's/^define(\[ucx_ver_major\], *\([0-9]*\)).*/\1/p' configure.ac)
minor=$(sed -n 's/^define(\[ucx_ver_minor\], *\([0-9]*\)).*/\1/p' configure.ac)
if [ -z "${major}" ] || [ -z "${minor}" ]; then
    echo "Cannot read the UCX version from configure.ac"
    exit 1
fi
current="v${major}.${minor}.x"

heads=$(git ls-remote --heads "${BUILD_REPOSITORY_URI}" 'v*.x')
legs=$(echo "${heads}" | sed 's#.*refs/heads/##' |
       grep -E '^v[0-9]+\.[0-9]+\.x$' |
       awk -F'[v.]' -v major="${major}" -v minor="${minor}" \
           '$2 < major || ($2 == major && $3 < minor)' |
       sort -V | tail -n "${num_releases}" |
       sed 's/^v\([0-9]*\)\.\([0-9]*\)\.x$/"ucx_\1_\2":{"ucx_tag":"&"}/' |
       paste -sd, -)
if [ -z "${legs}" ]; then
    echo "No release branch older than ${current} in ${BUILD_REPOSITORY_URI}"
    exit 1
fi

echo "Wire-compat targets for ${current}: {${legs}}"
# Split ##vso so a set -x trace is not parsed by the agent
printf '##%s[task.setvariable variable=matrix;isOutput=true]{%s}\n' vso "${legs}"
