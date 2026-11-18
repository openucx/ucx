#!/bin/sh
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Prints the container's cgroup CPU and memory counters.
p=POD_USAGE
c=/sys/fs/cgroup
now=$(date +%s)
start=$(stat -c %Y /proc/1 2>/dev/null)
echo "$p host=$(hostname) now=$now container_start=$start nproc=$(nproc 2>&1)" \
     "nproc_all=$(nproc --all 2>&1) lscpu=$(lscpu -p 2>/dev/null | grep -vc '^#')" \
     "NPROC=${NPROC:-unset} parallel_jobs=${parallel_jobs:-unset}"
if [ -f $c/cgroup.controllers ]; then
    echo "$p cgroup=v2 cpu.max=$(cat $c/cpu.max 2>&1) memory.max=$(cat $c/memory.max 2>&1)" \
         "memory.peak=$(cat $c/memory.peak 2>&1) memory.current=$(cat $c/memory.current 2>&1)"
    echo "$p cpu.stat $(tr '\n' ' ' < $c/cpu.stat 2>&1)"
    echo "$p memory.events $(tr '\n' ' ' < $c/memory.events 2>&1)"
    echo "$p memory.stat $(grep -E '^(anon|file|shmem) ' $c/memory.stat 2>&1 | tr '\n' ' ')"
    echo "$p cpu.pressure $(tr '\n' ' ' < $c/cpu.pressure 2>&1)"
else
    m=$c/memory
    for d in $c/cpu,cpuacct $c/cpuacct $c/cpu; do [ -f $d/cpuacct.usage ] && a=$d; [ -f $d/cpu.stat ] && s=$d; done
    echo "$p cgroup=v1 cpu.cfs_quota_us=$(cat $s/cpu.cfs_quota_us 2>&1) cpu.cfs_period_us=$(cat $s/cpu.cfs_period_us 2>&1)" \
         "memory.limit_in_bytes=$(cat $m/memory.limit_in_bytes 2>&1)" \
         "memory.max_usage_in_bytes=$(cat $m/memory.max_usage_in_bytes 2>&1)" \
         "memory.usage_in_bytes=$(cat $m/memory.usage_in_bytes 2>&1) memory.failcnt=$(cat $m/memory.failcnt 2>&1)"
    echo "$p cpuacct.usage_ns=$(cat $a/cpuacct.usage 2>&1)"
    echo "$p cpu.stat $(tr '\n' ' ' < $s/cpu.stat 2>&1)"
fi
ls $c 2>&1 | tr '\n' ' ' | sed "s/^/$p ls /"; echo
