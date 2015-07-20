#!/bin/sh

rm -rf autom4te.cache
mkdir -p config/m4 config/aux 
autoreconf -v --install || exit 1
rm -rf autom4te.cache
