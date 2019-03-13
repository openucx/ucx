#!/bin/bash -eE


PACKAGE=ucx
WS=$PWD
rpmspec=${PACKAGE}.spec
rpmmacros="--define='_rpmdir ${WS}/rpm-dist' --define='_srcrpmdir ${WS}/rpm-dist' --define='_sourcedir ${WS}' --define='_specdir ${WS}' --define='_builddir ${WS}'"
rpmopts="--nodeps --buildroot='${WS}/_rpm'"



opt_tarball=0
opt_srcrpm=0
opt_binrpm=0
opt_no_dist=0
defines=""

while test "$1" != ""; do
    case $1 in
        --tarball|-t) opt_tarball=1 ;;
        --srcrpm|-s)  opt_srcrpm=1 ;;
        --binrpm|-b)  opt_binrpm=1 ;;
        --no-dist)    opt_no_dist=1 ;;
        --define|-d)  defines="$defines --define '$2'"; shift ;;
        *)
            cat <<EOF
Unrecognized argument: $1

Valid arguments:

--tarball|-t        Create tarball
--srcrpm|-s         Create src.rpm
--binrpm|-b         Create bin.rpm
--no-dist           Undefine %{dist} tag
--define|-d <arg>   Add a define to rpmbuild


EOF
            exit 1
            ;;
    esac
    shift
done

if [ $opt_no_dist -eq 1 ]; then
    rpmmacros="$rpmmacros '--undefine=dist'"
fi

mkdir -p rpm-dist

if [ $opt_tarball -eq 1 ]; then
    make dist
fi

# Version includes revision, while tarball in Source doesn't have it since
# it uses GitHub standart name v<Version>.tar.gz, so make:
# ucx-1.3.0.6a61458.tar.gz --> v1.3.0.tar.gz for rpmbuild
tgz=(ucx*.tar.gz)
tarball=${tgz[0]}
link_tarball=$(perl -e '$fname=$ARGV[0]; ($new_name=$fname)=~s/^.+-(\d+\.\d+\.\d+)/v$1/; print $new_name' $tarball)
rm -f $link_tarball
ln -s $tarball $link_tarball

if [ $opt_srcrpm -eq 1 ]; then
    echo rpmbuild -bs $rpmmacros $rpmopts $rpmspec $defines | bash -eEx
fi

if [ $opt_binrpm -eq 1 ]; then
	# read build configuration
	source contrib/rpmdef.sh || exit 1

	with_arg() {
		module=$1
		with_arg=${2:-$module}
		if echo ${build_modules} | tr ':' '\n' | grep -q "^${module}$"
		then
			echo "--with ${with_arg}"
		else
			echo "--without ${with_arg}"
		fi
	}

	with_args=""
	with_args+=" $(with_arg cma)"
	with_args+=" $(with_arg cuda)"
	with_args+=" $(with_arg gdrcopy)"
	with_args+=" $(with_arg ib)"
	with_args+=" $(with_arg cm ib_cm)"
	with_args+=" $(with_arg knem)"
	with_args+=" $(with_arg rdmacm)"
	with_args+=" $(with_arg rocm)"
	with_args+=" $(with_arg ugni)"
	with_args+=" $(with_arg xpmem)"

	echo rpmbuild -bb $rpmmacros $rpmopts $rpmspec $defines $with_args | bash -eEx
fi

