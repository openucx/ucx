#!/bin/sh

rm -rf autom4te.cache
mkdir -p config/m4 config/aux
git submodule foreach ls
git submodule update --init --recursive
git submodule foreach ls
git submodule --quiet foreach ls configure.m4 || \
git submodule foreach git reset --hard HEAD
autoreconf -v --install || exit 1
rm -rf autom4te.cache
