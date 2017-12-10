#!/usr/bin/python
#
# Copyright (C) Mellanox Technologies Ltd. 2017-.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

import sys
import subprocess
import os
import re

#expected AM transport selections per given number of eps
mlx4_am = {
       2 :      "rc",
      16 :      "rc",
      32 :      "rc",
      64 :      "rc",
     256 :      "ud",
    1024 :      "ud",
 1000000 :      "ud",
}

mlx5_am = {
       2 :      "rc_mlx5",
      16 :      "rc_mlx5",
      32 :      "rc_mlx5",
      64 :      "dc_mlx5",
     256 :      "dc_mlx5",
    1024 :      "dc_mlx5",
 1000000 :      "dc_mlx5",
}

mlx5_am_roce = {
       2 :      "rc_mlx5",
      16 :      "rc_mlx5",
      32 :      "rc_mlx5",
      64 :      "rc_mlx5",
     256 :      "ud_mlx5",
    1024 :      "ud_mlx5",
 1000000 :      "ud_mlx5",
}

am_tls = {
    "mlx4"      : mlx4_am,
    "mlx5"      : mlx5_am,
    "mlx5_roce" : mlx5_am_roce
}

def find_am_transport(dev, neps) :

    ucx_info = bin_prefix+"/ucx_info -e -u t"

    os.putenv("UCX_TLS", "ib")
    os.putenv("UCX_NET_DEVICES", dev)

    output = subprocess.check_output(ucx_info + " -n " + str(neps) + " | grep am", shell=True)
    #print output

    match  = re.search(r'\d+:(\S+)/\S+', output)
    am_tls = match.group(1)

    #print am_tls
    return am_tls


if len(sys.argv) > 1:
    bin_prefix = sys.argv[1] + "/bin"
else:
    bin_prefix = "./src/tools/info"

dev_list = subprocess.check_output("ibstat -l", shell=True).splitlines()
port = "1"

for dev in sorted(dev_list):
    dev_attrs = subprocess.check_output("ibstat " + dev + " " + port, shell=True)
    if dev_attrs.find("State: Active") == -1:
        continue
    
    if dev_attrs.find("Link layer: Ethernet") == -1:
        dev_tl_map = am_tls[dev[0:dev.index('_')]]
    else:
        dev_tl_map = am_tls[dev[0:dev.index('_')]+"_roce"]

    for n_eps in sorted(dev_tl_map):
        tl = find_am_transport(dev + ':' + port, n_eps)
        print dev+':' + port + " eps: ", n_eps, " expected am tl: " + dev_tl_map[n_eps] + " selected: " + tl

        if dev_tl_map[n_eps] != tl:
            sys.exit(1)

sys.exit(0)

