#!/bin/bash -eEu
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Prepare one commit after PR has been approved.
#

message=
parent=
remote=

usage() {
    echo "usage: $0 -m <commit_message> [-r <remote_branch>] [-p <new_parent>]" >&2
    echo "" >&2
    echo "Arguments" >&2
    echo "  -m <commit_message>     commit message to use" >&2
    echo "  -r <remote_branch>      upstream master branch to merge into" >&2
    echo "  -p <new_parent>         new base to use for squashed commit" >&2
    echo "" >&2
    echo "Sample" >&2
    echo "    $0 -m \"Commit message\" -r upstream/master" >&2
    echo "    $0 -m \"Commit message\" -p adef30" >&2
    exit 1
}

param_checks() {
    if [ -z "$message" ]
    then
        usage
    fi

    if [ -z "$parent" ] && [ -z "$remote" ]
    then
        echo "error: parent must be specified when remote is missing (-p)" >&2
        exit 1
    fi

    if [ -z "$parent" ]
    then
        parent=$(git merge-base HEAD "$remote")
    fi

    object=$(git cat-file -t "$parent" || :)

    if [ "$object" != "commit" ]
    then
        echo "error: parent sha1 $parent is not a commit" >&2
        exit 1
    fi
}

while getopts "hm:r:p:" o; do
    case "$o" in
    m)
        message="$OPTARG"
        ;;
    r)
        remote="$OPTARG"
        ;;
    p)
        parent="$OPTARG"
        ;;
    *)
        usage
        ;;
    esac
done

param_checks

revision=$(git rev-parse HEAD)
tree=$(git rev-parse "$revision"^{tree})
commit=$(git commit-tree "$tree" -m "$message" -p "$parent")

git reset "$commit"
git log -2 --stat
