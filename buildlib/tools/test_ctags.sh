#!/usr/bin/env bash
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

set -eu

./contrib/ctags.sh

[[ -f tags ]] || { echo "TAG file not found"; exit 1; }

set +e
set -x

failures=

for tag in \
    tag_not_found \
    ucp_tag_send_nbx \
    ucs_error \
    ucp_request_free \
    test_ucp_am;
do
    echo checking tag=$tag
    vim -u NONE -c "set tags=./tags" -c "tag $tag" -c 'if v:errmsg != "" | cquit | else | quit | endif'
    if [[ $? -ne 0 ]]; then
        failures="$failures|$tag"
    fi
done

if [[ "$failures" = "|tag_not_found" ]]
then
    echo "SUCCESS"
    exit 0
else
    echo "FAILURES: ${failures}"
    exit 1
fi
