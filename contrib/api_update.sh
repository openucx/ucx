#!/usr/bin/env bash
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

set -euE

branch="${1?Specify remote version to use like "upstream/v1.18.x"}"
remote="${branch%%/*}"
version="$(echo "${branch##*/}" | sed -e 's@\(v[0-9]\+\.[0-9]\+\).*@\1@')"

if ! git diff-index --quiet HEAD
then
    echo "Current tree is not clean"
    exit 1
fi

if ! git rev-parse --verify --quiet "$branch" >/dev/null
then
    echo "Branch \"$branch\" does not exist"
    exit 1
fi

echo "Proceed with clean checkout of $branch (${version})?"
read -r -p "press <enter>"

set -x

if grep -qi "debian\|ubuntu" /etc/os-release 2>/dev/null
then
    sudo apt-get -y install doxygen doxygen-latex
else
    sudo yum -y install doxygen doxygen-latex
fi

git checkout "$branch"
git clean -xdf
git reset --hard

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
