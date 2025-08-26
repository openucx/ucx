#!/bin/sh -eE
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2025. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# This script checks that all installed headers are self-contained.
# Usage:
#        ./check_inst_headers.sh [include-install-dir]
#

CC=${CC:-gcc}
CXX=${CXX:-g++}
NVCC=${NVCC:-nvcc}

cd ${1:-.}

for filename in $(find -type f -name '*.h')
do
	# strip leading ./
	hfile=$(echo ${filename} | sed -e 's:^./::g')

	# skip some files which are documented to not be included directly
	if test "${hfile}" = "uct/api/tl.h"
	then
		continue
	fi

	# devices files should be compiled by nvcc
	file=$(basename ${hfile})
	if test "$file" != "${file#ucp_device_}"
	then
		if ! $NVCC --version >/dev/null 2>&1; then
			echo "SKIPPED $hfile (NVCC)"
			continue
		fi

		TMP=tmp.cu
		echo "#include <$hfile>" >$TMP
		$NVCC -I. -c $TMP -o /dev/null || { rm $TMP; exit 1; }
		rm $TMP
		echo "OK $hfile (NVCC)"
		continue
	fi

	# try to compile a test program (from stdin) which includes hfile
	for compile in "${CC} -Werror=strict-prototypes -x c" "${CXX} -x c++"
	do
		${compile} -I. -c - -o /dev/null -DHAVE_CONFIG_H=1 <<EOF
#include "${hfile}"
EOF
	done

	echo "OK $hfile"
done
