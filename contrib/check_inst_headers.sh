#!/bin/sh
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# This script checks that all installed headers are self-contained.
# Usage:
#        ./check_inst_headers.sh [include-install-dir]
#

CC=${CC:-gcc}

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
	${CC} -I. -x c -c - -o /dev/null -DHAVE_CONFIG_H=1 <<EOF || exit $?
#include "${hfile}"
EOF

	echo "OK $hfile"
done
