#!/bin/bash -eE
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


PACKAGE=ucx
WS=$PWD
rpmspec=${PACKAGE}.spec

rpmmacros="--define='_rpmdir ${WS}/rpm-dist' --define='_srcrpmdir ${WS}/rpm-dist' --define='_sourcedir ${WS}' --define='_specdir ${WS}' --define='_builddir ${WS}'"
rpmopts="--buildroot='${WS}/_rpm'"


opt_tarball=0
opt_srcrpm=0
opt_binrpm=0
opt_no_dist=0
opt_no_deps=0
opt_strict_ibverb_dep=0
opt_dbgrpm=0
defines=""

while test "$1" != ""; do
    case $1 in
        --tarball|-t) opt_tarball=1 ;;
        --srcrpm|-s)  opt_srcrpm=1 ;;
        --binrpm|-b)  opt_binrpm=1 ;;
        --dbgrpm|-d)  opt_dbgrpm=1 ;;
        --no-dist)    opt_no_dist=1 ;;
        --nodeps)     opt_no_deps=1 ;;
        --noclean)    rpmopts="$rpmopts --noclean" ;;
        --define|-d)  defines="$defines --define '$2'"; shift ;;
        --strict-ibverbs-dep) opt_strict_ibverb_dep=1 ;;
        *)
            cat <<EOF
Unrecognized argument: $1

Valid arguments:

--tarball|-t        Create tarball
--srcrpm|-s         Create src.rpm
--binrpm|-b         Create bin.rpm
--dbgrpm|-d         Create bin.rpm with debug function
--no-dist           Undefine %{dist} tag
--nodeps            Ignore build-time dependencies
--define|-d <arg>   Add a define to rpmbuild
--strict-ibverbs-dep Add RPM "Requires: libibverbs == VER-RELEASE" (libibverbs has to be installed)

EOF
            exit 1
            ;;
    esac
    shift
done

if [ $opt_no_dist -eq 1 ]; then
    rpmmacros="$rpmmacros '--undefine=dist'"
fi

if [ $opt_strict_ibverb_dep -eq 1 ]; then
    libibverbs_ver=$(rpm -q libibverbs --qf '%{version}-%{release}')
    rpmmacros="${rpmmacros} --define='extra_deps libibverbs == ${libibverbs_ver}'"
fi

if [ $opt_dbgrpm -eq 1 ]; then
    rpmmacros="--define 'debug 1' ${rpmmacros}"
fi

if [ $opt_no_deps -eq 1 ]; then
    rpmopts="$rpmopts --nodeps"
fi

mkdir -p rpm-dist

if [ $opt_tarball -eq 1 ]; then
    make dist
fi

# Version includes revision, while tarball in Source doesn't have it since
# it uses GitHub standard name v<Version>.tar.gz, so make:
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
		if (echo ${build_modules}  | tr ':' '\n' | grep -q "^${module}$") ||
		   (echo ${build_bindings} | tr ':' '\n' | grep -q "^${module}$")
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
	with_args+=" $(with_arg knem)"
	with_args+=" $(with_arg rdmacm)"
	with_args+=" $(with_arg rocm)"
	with_args+=" $(with_arg ugni)"
	with_args+=" $(with_arg xpmem)"
	with_args+=" $(with_arg fuse)"
	with_args+=" $(with_arg mad)"
	with_args+=" $(with_arg mlx5)"
	with_args+=" $(with_arg efa)"

	echo rpmbuild -bb $rpmmacros $rpmopts $rpmspec $defines $with_args | bash -eEx
fi
