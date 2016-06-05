#!/bin/sh -eEx
#
# Run this from UCX build directory
#

rev=$(git rev-parse --short HEAD)
make docs
mkdir -p ucx.wiki
cd ucx.wiki
git init .
[ -d .git/refs/remotes/origin ] || git remote add origin https://github.com/openucx/ucx.wiki.git
git pull
cp -f ../doc/doxygen-doc/ucx.pdf ./
git ci ucx.pdf -m "update ucx.pdf for $rev"
git push
