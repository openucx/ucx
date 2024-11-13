#!/usr/bin/env bash
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

source ./buildlib/tools/codestyle.sh

ucx=://github.com/openucx/ucx
upstream=$(git remote -v  | grep -P "[\t ].*${ucx}.*[\t ].*fetch" | cut -f 1 | head -n 1)
base=$(git merge-base "$upstream"/master HEAD)

if ! git diff-index --quiet HEAD; then
    echo "error: tree not clean"
    exit 1
fi

codestyle_check_commit_title "$base"..HEAD
if [[ $? -ne 0 ]]
then
    echo "error: fix commit title"
    exit 1
fi

# Code format
module load dev/llvm-ucx || :
git clang-format --diff "$base" HEAD | patch -p1
module unload dev/llvm-ucx || :

# Codespell
codestyle_check_spell --write-changes || :

# Create commit for pre-push fixes
if ! git diff-index --quiet HEAD; then
    git add -p
    # returns 1 if cached index is not empty
    git diff --cached --exit-code || git commit -m "$title"
fi

# Pushing
if [ "${1-}" = "--push" ]
then

    opt="${2-}"
    remote="${opt%%/*}"
    branch="${opt#*/}"

    if [ "$remote" = "$opt" ] || [ -z "$remote" ] || [ -z "$branch" ]
    then
        echo "error: specify push location with '--push <remote>/<branch_name>'"
        exit 1
    fi

    cmd="git push $remote HEAD:refs/heads/$branch"
    echo "$cmd"
    echo "<enter> or <ctrl-c> to abort"
    read -r
    $cmd
fi
