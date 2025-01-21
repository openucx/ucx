#!/bin/sh -eE
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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

	# try to compile a test program (from stdin) which includes hfile
	for compile in "${CC} -x c" "${CXX} -x c++"
	do
		${compile} -I. -c - -o /dev/null -DHAVE_CONFIG_H=1 <<EOF
#include "${hfile}"
EOF
	done

	echo "OK $hfile"
done
