#!/bin/sh

usage()
{
	echo "Usage: autogen.sh <options>"
	echo
	echo "  -h|--help         Show this help message"
	echo
}

for key in "$@"
do
	case $key in
	-h|--help)
		usage
		exit 0
		;;
	*)
		usage
		exit -2
		;;
	esac
done

rm -rf autom4te.cache
mkdir -p config/m4 config/aux

git submodule update --init src/uct/ib/mlx5/gdaki/gpunetio

autoreconf -v --install || exit 1
rm -rf autom4te.cache
