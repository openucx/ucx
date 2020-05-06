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
import commands
from distutils.version import LooseVersion
from optparse import OptionParser


#expected AM transport selections per given number of eps
mlx4_am = {
       2 :      "rc_verbs",
      16 :      "rc_verbs",
      32 :      "rc_verbs",
      64 :      "rc_verbs",
     256 :      "ud_verbs",
     512 :      "ud_verbs",
    1024 :      "ud_verbs",
 1000000 :      "ud_verbs",
}

mlx5_am = {
       2 :      "rc_mlx5",
      16 :      "rc_mlx5",
      32 :      "rc_mlx5",
      64 :      "dc_mlx5",
     256 :      "dc_mlx5",
     512 :      "dc_mlx5",
    1024 :      "dc_mlx5",
 1000000 :      "dc_mlx5",
}

mlx5_am_no_dc = {
       2 :      "rc_mlx5",
      16 :      "rc_mlx5",
      32 :      "rc_mlx5",
      64 :      "rc_mlx5",
     256 :      "ud_mlx5",
     512 :      "ud_mlx5",
    1024 :      "ud_mlx5",
 1000000 :      "ud_mlx5",
}

# check that UCX_NUM_EPS work
mlx5_am_override = {
       2 :      "rc_mlx5",
      16 :      "rc_mlx5",
      32 :      "rc_mlx5",
      64 :      "rc_mlx5",
     256 :      "rc_mlx5",
     512 :      "rc_mlx5",
    1024 :      "rc_mlx5",
 1000000 :      "rc_mlx5",
}

mlx4_am_override = {
       2 :      "rc_verbs",
      16 :      "rc_verbs",
      32 :      "rc_verbs",
      64 :      "rc_verbs",
     256 :      "rc_verbs",
     512 :      "rc_verbs",
    1024 :      "rc_verbs",
 1000000 :      "rc_verbs",
}

am_tls =  {
    "mlx4"            : mlx4_am,
    "mlx5"            : mlx5_am,
    "mlx5_roce_dc"    : mlx5_am,       # mlx5 RoCE port which supports DC
    "mlx5_roce_no_dc" : mlx5_am_no_dc, # mlx5 RoCE port which doesn't support DC
    "mlx4_override"   : mlx4_am_override,
    "mlx5_override"   : mlx5_am_override
}

def exec_cmd(cmd):
    if options.verbose:
        print cmd

    status, output = commands.getstatusoutput(cmd)
    if options.verbose:
        print "return code " + str(status)
        print output

    return status, output

def find_am_transport(dev, neps, override = 0) :

    os.putenv("UCX_TLS", "ib")
    os.putenv("UCX_NET_DEVICES", dev)

    if (override):
        os.putenv("UCX_NUM_EPS", "2")

    status, output = exec_cmd(ucx_info + ucx_info_args + str(neps) + " | grep am")

    os.unsetenv("UCX_TLS")
    os.unsetenv("UCX_NET_DEVICES")

    match = re.search(r'\d+:(\S+)/\S+', output)
    if match:
        am_tls = match.group(1)
        if (override):
            os.unsetenv("UCX_NUM_EPS")

        return am_tls
    else:
        return "no am tls"

def test_fallback_from_rc(dev, neps) :

    os.putenv("UCX_TLS", "ib")
    os.putenv("UCX_NET_DEVICES", dev)

    status,output = exec_cmd(ucx_info + ucx_info_args + str(neps) + " | grep rc")

    os.unsetenv("UCX_TLS")
    os.unsetenv("UCX_NET_DEVICES")

    if output != "":
        print "RC transport must not be used when estimated number of EPs = " + str(neps)
        sys.exit(1)

    os.putenv("UCX_TLS", "rc,ud,tcp")

    status,output_rc = exec_cmd(ucx_info + ucx_info_args + str(neps) + " | grep rc")

    status,output_tcp = exec_cmd(ucx_info + ucx_info_args + str(neps) + " | grep tcp")

    if output_rc != "" or output_tcp != "":
        print "RC/TCP transports must not be used when estimated number of EPs = " + str(neps)
        sys.exit(1)

    os.unsetenv("UCX_TLS")

parser = OptionParser()
parser.add_option("-p", "--prefix", metavar="PATH", help = "root UCX directory")
parser.add_option("-v", "--verbose", action="store_true", \
                  help = "verbose output", default=False)
(options, args) = parser.parse_args()

if options.prefix == None:
    bin_prefix = "./src/tools/info"
else:
    bin_prefix = options.prefix + "/bin"

if not (os.path.isdir(bin_prefix)):
    print "directory \"" + bin_prefix + "\" does not exist"
    parser.print_help()
    exit(1)

ucx_info = bin_prefix + "/ucx_info"
ucx_info_args = " -e -u t -n "

status, output = exec_cmd(ucx_info + " -c | grep -e \"UCX_RC_.*_MAX_NUM_EPS\"")
match = re.findall(r'\S+=(\d+)', output)
if match:
    rc_max_num_eps = int(max(match))
else:
    rc_max_num_eps = 0

status, output = exec_cmd("ibv_devinfo  -l | tail -n +2 | sed -e 's/^[ \t]*//' | head -n -1 ")
dev_list = output.splitlines()
port = "1"

for dev in sorted(dev_list):
    status, dev_attrs = exec_cmd("ibv_devinfo -d " + dev + " -i " + port)
    if dev_attrs.find("PORT_ACTIVE") == -1:
        continue

    driver_name = os.path.basename(os.readlink("/sys/class/infiniband/%s/device/driver" % dev))
    dev_name    = driver_name.split("_")[0] # should be mlx4 or mlx5
    if not dev_name in ['mlx4', 'mlx5']:
        print "Skipping unknown device: ", dev_name
        continue

    if dev_attrs.find("Ethernet") == -1:
        dev_tl_map = am_tls[dev_name]
        dev_tl_override_map = am_tls[dev_name + "_override"]
        override = 1
    else:
        fw_ver = open("/sys/class/infiniband/%s/fw_ver" % dev).read()
        if LooseVersion(fw_ver) >= LooseVersion("16.23.0"):
            dev_tl_map = am_tls[dev_name+"_roce_dc"]
        else:
            dev_tl_map = am_tls[dev_name+"_roce_no_dc"]
        override = 0

    for n_eps in sorted(dev_tl_map):
        tl = find_am_transport(dev + ':' + port, n_eps)
        print dev+':' + port + "               eps: ", n_eps, " expected am tl: " + \
              dev_tl_map[n_eps] + " selected: " + tl

        if dev_tl_map[n_eps] != tl:
            sys.exit(1)

        if override:
            tl = find_am_transport(dev + ':' + port, n_eps, 1)
            print dev+':' + port + " UCX_NUM_EPS=2 eps: ", n_eps, " expected am tl: " + \
                  dev_tl_override_map[n_eps] + " selected: " + tl

            if dev_tl_override_map[n_eps] != tl:
                sys.exit(1)

        if n_eps >= (rc_max_num_eps * 2):
            test_fallback_from_rc(dev + ':' + port, n_eps)

sys.exit(0)

