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

if [ $# -ne 1 ]; then
    echo "Usage: $0 <num_releases>"
    exit 1
fi

num_releases=$1

major=$(sed -n 's/^define(\[ucx_ver_major\], *\([0-9]*\)).*/\1/p' configure.ac)
minor=$(sed -n 's/^define(\[ucx_ver_minor\], *\([0-9]*\)).*/\1/p' configure.ac)
if [ -z "${major}" ] || [ -z "${minor}" ]; then
    echo "Cannot read the UCX version from configure.ac"
    exit 1
fi
current="v${major}.${minor}.x"

heads=$(git ls-remote --heads "${BUILD_REPOSITORY_URI}" 'v*.x')
branches=$(echo "${heads}" | sed 's#.*refs/heads/##' |
           grep -E '^v[0-9]+\.[0-9]+\.x$' | sort -V)

releases=""
for branch in ${branches}; do
    # sort -C succeeds when branch <= current
    if [ "${branch}" != "${current}" ] &&
       printf '%s\n%s\n' "${branch}" "${current}" | sort -CV; then
        releases+="${branch}"$'\n'
    fi
done

releases=$(printf '%s' "${releases}" | tail -n "${num_releases}")
if [ -z "${releases}" ]; then
    echo "No release branch older than ${current} in ${BUILD_REPOSITORY_URI}"
    exit 1
fi

legs=""
for ref in ${releases}; do
    IFS=. read -r major minor _ <<< "${ref#v}"
    legs+="${legs:+,}\"ucx_${major}_${minor}\":{\"ucx_tag\":\"${ref}\"}"
done

echo "Wire-compat targets for ${current}: {${legs}}"
# Split ##vso so a set -x trace is not parsed by the agent
printf '##%s[task.setvariable variable=matrix;isOutput=true]{%s}\n' vso "${legs}"
