#!/usr/bin/env bash
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

codestyle_check_commit_title() {
    local range="$1"
    local err=0

    for sha1 in `git log "$range" --format="%h"`
    do
        title=`git log -1 --format="%s" $sha1`
        if echo $title | grep -qP '^Merge |^[0-9A-Z/_\-]*: \w'
        then
            echo "Good commit title: '$title'"
        else
            echo "Bad commit title: '$title'"
            err=1
        fi
    done

    return $err
}

codestyle_check_spell() {
    python3 -m venv /tmp/codespell_env
    source /tmp/codespell_env/bin/activate
    pip3 install codespell
    codespell "$@"
}
