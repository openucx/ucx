#!/usr/bin/env bash
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

base="$(realpath $0)"
cd "$(dirname $base)"/..

echo "Using $PWD"
if [[ ! -e autogen.sh ]]
then
    echo "Failed to get root of directory from '$0'"
    return 1
fi

rm -f tags || :
ctags -R -f tags .

find . -name "*.inl" -type f -exec \
    ctags -R -f tags \
    --append=yes \
    --language-force=C \
    {} \;

find . -type f -and \( -name '*.inl' -or -name '*.c' \) \
    -exec awk -f ./contrib/ctags_ucx.awk {} \; >>tags

LC_COLLATE=C sort -o tags{,}
