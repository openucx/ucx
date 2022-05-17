#!/bin/sh
#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

# check_tls.sh [extra-tls-to-check]

res=true;
for run in ./uct_info_static ./uct_info; do
    echo Check for $run;
    output=`$run`;
    for i in tcp self sysv posix $@; do
        if `echo $output | grep -q "Transport: $i"`; then
            echo $i... ok;
        else
            res=false;
            echo $i... fail;
        fi;
    done;
    if test $res = false; then
        echo failed to check tl in $run;
        break;
    fi;
done;
$res

# done
