#!/bin/sh

rm -rf autom4te.cache
mkdir -p config/m4 config/aux external/jemalloc

pushd external/jemalloc
autoconf
popd

autoreconf -v --install --no-recursive || exit 1
rm -rf autom4te.cache
