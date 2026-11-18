#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#
# Set the wire-compat test matrix as the "matrix" output variable: the extra
# refs, plus every vX.Y.x release branch starting from <min_release>.
#
# Usage: wire_compat_matrix.sh <min_release> [extra_ref...]
# Env:   BUILD_REPOSITORY_URI - repository to take the release branches from
#

if [ $# -lt 1 ]; then
    echo "Usage: $0 <min_release> [extra_ref...]"
    exit 1
fi

min_release=$1
shift

heads=$(git ls-remote --heads "${BUILD_REPOSITORY_URI}" 'v*.x')
branches=$(echo "${heads}" | sed 's#.*refs/heads/##' |
           grep -E '^v[0-9]+\.[0-9]+\.x$' | sort -V)

releases=""
for branch in ${branches}; do
    # sort -C succeeds when min_release <= branch
    if printf '%s\n%s\n' "${min_release}" "${branch}" | sort -CV; then
        releases+=" ${branch}"
    fi
done

if [ -z "${releases}" ]; then
    echo "No release branch >= ${min_release} in ${BUILD_REPOSITORY_URI}"
    exit 1
fi

legs=""
for ref in "$@" ${releases}; do
    IFS=. read -r major minor _ <<< "${ref#v}"
    legs+="${legs:+,}\"ucx_${major}_${minor}\":{\"ucx_tag\":\"${ref}\"}"
done

echo "Wire-compat targets: {${legs}}"
# Split ##vso so a set -x trace is not parsed by the agent
printf '##%s[task.setvariable variable=matrix;isOutput=true]{%s}\n' vso "${legs}"
