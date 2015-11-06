#!/bin/sh

rm -rf autom4te.cache
mkdir -p config/m4 config/aux external/jemalloc

GIT_FOUND=1
command git --help 2>&1 > /dev/null || { \
    echo "Git not found, will not retrieve jemalloc"; \
    GIT_FOUND=0; }

if [ $GIT_FOUND -eq 1 ]
then
git submodule init
git submodule update
fi

if [ -e external/jemalloc/configure.ac ]
then
pushd external/jemalloc
autoconf || exit 2
popd
fi

autoreconf -v --install --no-recursive || exit 1
rm -rf autom4te.cache
