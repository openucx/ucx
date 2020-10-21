#!/bin/sh

usage()
{
	echo "Usage: autogen.sh <options>"
	echo
	echo "  -h|--help         Show this help message"
	echo "  --with-ucg        Fetch UCG submodule"
	echo
}

with_ucg="no"

for key in "$@"
do
	case $key in
	-h|--help)
		usage
		exit 0
		;;
	--with-ucg)
		with_ucg="yes"
		;;
	*)
		usage
		exit -2
		;;
	esac
done

rm -rf autom4te.cache
mkdir -p config/m4 config/aux

if [ "X$with_ucg" = "Xyes" ]
then
	git submodule update --init --recursive --remote
fi

autoreconf -v --install || exit 1
rm -rf autom4te.cache
