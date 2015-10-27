#!/bin/sh

rm -rf autom4te.cache
mkdir -p config/m4 config/aux external/jemalloc

if [ -e external/jemalloc/configure.ac ]
then
pushd external/jemalloc
autoconf || exit 2
popd
fi

autoreconf -v --install --no-recursive || exit 1
rm -rf autom4te.cache
