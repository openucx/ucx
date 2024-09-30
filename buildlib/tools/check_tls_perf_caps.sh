#!/bin/bash -eE
#
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

realdir="$(realpath "$(dirname "$0")")"
source "${realdir}"/../az-helpers.sh

legacy_info_file=ucx_info_legacy.txt
pr_info_file=ucx_info_pr.txt

export UCX_TCP_BRIDGE_ENABLE=y #bridge devices are not shown by default since v1.16

# Print legacy library versions and configurations
LD_LIBRARY_PATH=${UCX_LEGACY_PATH}/lib "${UCX_LEGACY_PATH}/bin/ucx_info" -v
LD_LIBRARY_PATH=${UCX_LEGACY_PATH}/lib "${UCX_LEGACY_PATH}/bin/ucx_info" -d > ${legacy_info_file}
echo ${legacy_info_file}:
cat ${legacy_info_file}

# Print new library versions and configurations
LD_LIBRARY_PATH=${UCX_PR_PATH}/lib "${UCX_PR_PATH}/bin/ucx_info" -v
LD_LIBRARY_PATH=${UCX_PR_PATH}/lib "${UCX_PR_PATH}/bin/ucx_info" -d > ${pr_info_file}
echo ${pr_info_file}:
cat ${pr_info_file}

res=true
for tl_name in $(grep Transport ${legacy_info_file} | awk '{print $3}')
do
    old_tl_caps=$(grep -A 8 "Transport: $tl_name" ${legacy_info_file} || true)
    new_tl_caps=$(grep -A 8 "Transport: $tl_name" ${pr_info_file}     || true)
    for device in  $(echo "${old_tl_caps}" | grep Device | awk '{print $3}')
    do
      old_caps=$(echo "$old_tl_caps" | grep -A 7 "Device: $device" || true)
      new_caps=$(echo "$new_tl_caps" | grep -A 7 "Device: $device" || true)
      for cap in bandwidth latency overhead
      do
        old_cap=$(echo "$old_caps" | grep $cap | sed -e "s/^[^:]*:[ \t]*//" || true)
        new_cap=$(echo "$new_caps" | grep $cap | sed -e "s/^[^:]*:[ \t]*//" || true)
        if [ "$old_cap" != "$new_cap" ]
        then
          azure_log_error "Fail: (${device}/${tl_name}/${cap}) ${old_cap} != ${new_cap}"
          res=false
        else
          echo "${device}/${tl_name}/${cap}: ${old_cap}"
        fi
       done
    done
done

$res
