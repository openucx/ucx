#!/bin/bash -euE
#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#


branch=${1:-""}
remote="${branch%%/*}"
version="$(echo "${branch##*/}" | sed -e 's@\(v[0-9]\+\.[0-9]\+\).*@\1@')"

if [ -z "$branch" ] || [ -z "$remote" ] || [ -z "$version" ]; then
    echo "usage: $0 <branch_name>" >&2
    echo "" >&2
    echo "Creates a commit ready to push on remote gh-pages branch" >&2
    echo "" >&2
    echo "Argument" >&2
    echo "  <branch_name>    Remote branch to create documentation for"
    echo "" >&2
    echo "Sample" >&2
    echo "  $0 upstream/v1.18.x" >&2
    exit 1
fi

if ! git rev-parse --verify --quiet "$branch" >/dev/null
then
    echo "Branch \"$branch\" does not exist"
    exit 1
fi

echo "Proceed with documentation generation for $branch (${version})?"
read -r -p "press <enter>"

set -x

if grep -qi "debian\|ubuntu" /etc/os-release 2>/dev/null
then
    cmd="dpkg -l"
else
    cmd="rpm -q"
fi

$cmd doxygen doxygen-latex || { echo "error: install doxygen and doxygen-latex packages"; exit 1; }

subdir=ucx_docs
rm -rf $subdir || :
git worktree prune
git worktree add $subdir "$branch"
cd $subdir

./autogen.sh
./configure --with-docs-only
make docs

git checkout "$remote"/gh-pages
mkdir api/"$version"
ln -snf "$version" api/latest

cp docs/doxygen-doc/ucx.pdf "api/$version/ucx-$version.pdf"
ln -s "ucx-$version.pdf" "api/$version/ucx.pdf"
cp -ar docs/doxygen-doc/html "api/$version/"

git add api/latest "api/$version"
git commit -m "add $version documentation"

git --no-pager show --stat
sha=$(git rev-parse HEAD)
cd -

echo "Push commit with: git push $remote $sha:gh-pages"
