#!/usr/bin/python2
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
import itertools
import contextlib
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

tl_aliases = {
    "mm":   ["posix", "sysv", "xpmem", ],
    "sm":   ["posix", "sysv", "xpmem", "knem", "cma", "rdmacm", "sockcm", ],
    "shm":  ["posix", "sysv", "xpmem", "knem", "cma", "rdmacm", "sockcm", ],
    "ib":   ["rc_verbs", "ud_verbs", "rc_mlx5", "ud_mlx5", "dc_mlx5", "rdmacm", ],
    "ud_v": ["ud_verbs", "rdmacm", ],
    "ud_x": ["ud_mlx5", "rdmacm", ],
    "ud":   ["ud_mlx5", "ud_verbs", "rdmacm", ],
    "rc_v": ["rc_verbs", "ud_verbs:aux", "rdmacm", ],
    "rc_x": ["rc_mlx5", "ud_mlx5:aux", "rdmacm", ],
    "rc":   ["rc_mlx5", "ud_mlx5:aux", "rc_verbs", "ud_verbs:aux", "rdmacm", ],
    "dc":   ["dc_mlx5", "rdmacm", ],
    "dc_x": ["dc_mlx5", "rdmacm", ],
    "ugni": ["ugni_smsg", "ugni_udt:aux", "ugni_rdma", ],
    "cuda": ["cuda_copy", "cuda_ipc", "gdr_copy", ],
    "rocm": ["rocm_copy", "rocm_ipc", "rocm_gdr", ],
}

@contextlib.contextmanager
def _override_env(var_name, value):
    if value is None:
        yield
        return

    prev_value = os.getenv(var_name)
    os.putenv(var_name, value)
    try:
        yield
    finally:
        os.putenv(var_name, prev_value) if prev_value else os.unsetenv(var_name)

def exec_cmd(cmd):
    if options.verbose:
        print cmd

    status, output = commands.getstatusoutput(cmd)
    if options.verbose:
        print "return code " + str(status)
        print output

    return status, output

def find_am_transport(dev, neps=1, override=0, tls="ib"):
    if (override):
        os.putenv("UCX_NUM_EPS", "2")
    
    with _override_env("UCX_TLS", tls), \
         _override_env("UCX_NET_DEVICES", dev):

        status, output = exec_cmd(ucx_info + ucx_info_args + str(neps) + " | grep am")

    match = re.search(r'\d+:(\S+)/\S+', output)
    if match:
        am_tls = match.group(1)
        if (override):
            os.unsetenv("UCX_NUM_EPS")

        return am_tls
    else:
        return None

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

def test_ucx_tls_positive(tls):
    # Use TLS list in "allow" mode and verify that the found tl is in the list
    found_tl = find_am_transport(None, tls=tls)
    print "Using UCX_TLS=" + tls + ", found TL: " + str(found_tl)
    if tls == 'all':
        return
    if not found_tl:
        sys.exit(1)
    tls = tls.split(',')
    if found_tl in tls or "\\" + found_tl in tls:
        return
    for tl in tls:
        if tl in tl_aliases and found_tl in tl_aliases[tl]:
            return
    print "Found TL doesn't belong to the allowed UCX_TLS"
    sys.exit(1)

def test_ucx_tls_negative(tls):
    # Use TLS list in "negate" mode and verify that the found tl is not in the list
    found_tl = find_am_transport(None, tls="^"+tls)
    print "Using UCX_TLS=^" + tls + ", found TL: " + str(found_tl)
    tls = tls.split(',')
    if not found_tl or found_tl in tls:
        print "No available TL found"
        sys.exit(1)
    for tl in tls:
        if tl in tl_aliases and found_tl in tl_aliases[tl]:
            print "Found TL belongs to the forbidden UCX_TLS"
            sys.exit(1)

def _powerset(iterable, with_empty_set=True):
    iterable_list = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(iterable_list, r) for r in \
            range(0 if with_empty_set else 1, len(iterable_list) + 1))

def test_tls_allow_list(ucx_info):
    status, output = exec_cmd(ucx_info + " -d | grep Transport | awk '{print $3}'")
    available_tls = set(output.splitlines())

    # Add some basic variants (those that are available on this platform)
    tls_variants = [tls_variant for tls_variant in ["tcp", "posix", "xpmem"] if \
                    tls_variant in available_tls]

    # Add some IB variant (both strict and alias), if available
    for tls_variant in available_tls:
        if tls_variant.startswith("rc_") or tls_variant.startswith("dc_") or \
           tls_variant.startswith("ud_"):
            tls_variants += ["ib", "\\" + tls_variant]
            break

    tls_variants = _powerset(tls_variants, with_empty_set=False)
    test_funcs = [test_ucx_tls_positive, test_ucx_tls_negative]
    for (tls_variant, test_func) in \
        itertools.product(tls_variants, test_funcs):
        test_func(",".join(tls_variant))

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

    if not os.path.exists("/sys/class/infiniband/%s/ports/%s/gids/0" % (dev, port)):
        print "Skipping dummy device: ", dev
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
              dev_tl_map[n_eps] + " selected: " + str(tl)

        if dev_tl_map[n_eps] != tl:
            sys.exit(1)

        if override:
            tl = find_am_transport(dev + ':' + port, n_eps, 1)
            print dev+':' + port + " UCX_NUM_EPS=2 eps: ", n_eps, " expected am tl: " + \
                  dev_tl_override_map[n_eps] + " selected: " + str(tl)

            if dev_tl_override_map[n_eps] != tl:
                sys.exit(1)

        if n_eps >= (rc_max_num_eps * 2):
            test_fallback_from_rc(dev + ':' + port, n_eps)

# Test UCX_TLS configuration (TL choice according to "allow" and "negate" lists)
test_tls_allow_list(ucx_info)

sys.exit(0)

