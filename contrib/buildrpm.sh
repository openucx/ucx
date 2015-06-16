#!/bin/bash -eE


PACKAGE=ucx
WS=$PWD
rpmspec=${PACKAGE}.spec
rpmmacros="--define='_rpmdir ${WS}/rpm-dist' --define='_srcrpmdir ${WS}/rpm-dist' --define='_sourcedir ${WS}' --define='_specdir ${WS}' --define='_builddir ${WS}'"
rpmopts="--nodeps --buildroot='${WS}/_rpm'"



opt_tarball=0
opt_srcrpm=0
opt_binrpm=0

while test "$1" != ""; do
    case $1 in
        --tarball|-t) opt_tarball=1 ;;
        --srcrpm|-s)  opt_srcrpm=1 ;;
        --binrpm|-b)  opt_binrpm=1 ;;
        *)
            cat <<EOF
Unrecognized argument: $1

Valid arguments:

--tarball|-t    Create tarball
--srcrpm|-s     Create src.rpm
--binrpm|-b     Create bin.rpm


EOF
            exit 1
            ;;
    esac
    shift
done


mkdir -p rpm-dist

if [ $opt_tarball -eq 1 ]; then
    make dist
fi

if [ $opt_srcrpm -eq 1 ]; then
    echo rpmbuild -bs $rpmmacros $rpmopts $rpmspec | bash -eEx
fi

if [ $opt_binrpm -eq 1 ]; then
    echo rpmbuild -bb $rpmmacros $rpmopts $rpmspec | bash -eEx
fi

