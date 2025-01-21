#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See file LICENSE for terms.
#

cd $(dirname $0)/..

echo "Using $PWD"

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
