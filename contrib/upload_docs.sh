#!/bin/sh -eEx
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Run this from UCX build directory.
#

rev=$(git rev-parse --short HEAD)
make docs
mkdir -p ucx.wiki
cd ucx.wiki
git init .
git remote show origin &>/dev/null || git remote add origin https://github.com/openucx/ucx.wiki.git
git fetch --all
git checkout -t origin/master -f
git pull
cp -f ../doc/doxygen-doc/ucx.pdf ./
git commit ucx.pdf -m "update ucx.pdf for $rev"
git push
