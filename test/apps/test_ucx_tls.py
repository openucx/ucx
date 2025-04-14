#!/usr/bin/env python3
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

import sys
import subprocess
import os
import re
import itertools
import contextlib
from optparse import OptionParser
from pkg_resources import parse_version


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
    "ib":   ["rc_verbs", "ud_verbs", "rc_mlx5", "ud_mlx5", "dc_mlx5", "rdmacm",
             "ud_mlx5:aux", "ud_verbs:aux", ],
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
def _override_env(env_vars):
    prev_values = []
    for var_name, value in env_vars:
        prev_values.append((var_name, os.getenv(var_name)))
        if value is not None:
            os.putenv(var_name, value)
        else:
            os.unsetenv(var_name)

    try:
        yield
    finally:
        for var_name, prev_value in prev_values:
            os.putenv(var_name, prev_value) if prev_value else os.unsetenv(var_name)

def exec_cmd(cmd):
    if options.verbose:
        print(cmd)

    status, output = subprocess.getstatusoutput(cmd)
    if options.verbose:
        print(f"return code {status}")
        print(output)

    return status, output

def find_transport(dev=None, neps=1, override=0, tls="ib", protocol="am"):
    if (override):
        os.putenv("UCX_NUM_EPS", "2")

    env_vars = [("UCX_TLS", tls)]

    # Set up environment variables based on protocol type
    if protocol == "am" and dev:
        env_vars.append(("UCX_NET_DEVICES", dev))

    # Use context manager for all environment variables
    with _override_env(env_vars):
        # Choose the appropriate arguments and grep pattern based on protocol type
        if protocol == "keepalive":
            args = ucx_info_eh_args
        elif protocol == "am":  # am transport
            args = ucx_info_args

        status, output = exec_cmd(f"{ucx_info}{args}{neps} | grep {protocol}")

    match = re.search(r'\d+:(\S+)/\S+', output)
    if match:
        proto_tls = match.group(1)
        if override:
            os.unsetenv("UCX_NUM_EPS")

        return proto_tls
    else:
        return None

def find_am_transport(dev, neps=1, override=0, tls="ib"):
    return find_transport(dev=dev, neps=neps, override=override,
                          tls=tls, protocol="am")

def test_fallback_from_rc(dev, neps) :

    os.putenv("UCX_TLS", "ib")
    os.putenv("UCX_NET_DEVICES", dev)

    status, output = exec_cmd(f"{ucx_info}{ucx_info_args}{neps} | grep rc")

    os.unsetenv("UCX_TLS")
    os.unsetenv("UCX_NET_DEVICES")

    if output != "":
        print(f"RC transport must not be used when estimated number of EPs = {neps}")
        sys.exit(1)

    os.putenv("UCX_TLS", "rc,ud,tcp")

    status, output_rc = exec_cmd(f"{ucx_info}{ucx_info_args}{neps} | grep rc")

    status, output_tcp = exec_cmd(f"{ucx_info}{ucx_info_args}{neps} | grep tcp")

    if output_rc != "" or output_tcp != "":
        print(f"RC/TCP transports must not be used when estimated number of EPs = {neps}")
        sys.exit(1)

    os.unsetenv("UCX_TLS")

def test_ucx_tls_positive(tls):
    # Use TLS list in "allow" mode and verify that the found tl is in the list
    found_tl = find_am_transport(None, tls=tls)
    print(f"Using UCX_TLS={tls}, found TL: {found_tl}")
    if tls == 'all':
        return
    if not found_tl:
        sys.exit(1)
    tls = tls.split(',')
    if found_tl in tls or f"\\{found_tl}" in tls:
        return
    for tl in tls:
        if tl in tl_aliases and found_tl in tl_aliases[tl]:
            return
    print("Found TL doesn't belong to the allowed UCX_TLS")
    sys.exit(1)

def test_ucx_tls_negative(tls, protocol="am", forbidden_tls=None):
    # Use TLS list in "negate" mode
    found_tl = find_transport(tls="^"+tls, protocol=protocol)
    print(f"Using UCX_TLS=^{tls}, found {protocol} TL: {found_tl}")
    if not found_tl:
        print("No available TL found")
        sys.exit(1)

    # If forbidden_tls is provided, verify that the found tl is not in that list
    if forbidden_tls is not None:
        if found_tl in forbidden_tls:
            print(f"Found forbidden TL: {found_tl}")
            sys.exit(1)
        return

    # Otherwise, check against the tls list
    tls = tls.split(',')
    if found_tl in tls:
        print(f"Found forbidden TL: {found_tl}")
        sys.exit(1)
    for tl in tls:
        if tl in tl_aliases and found_tl in tl_aliases[tl]:
            print(f"Found forbidden TL: {found_tl}")
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
        if tls_variant.startswith("dc_") or tls_variant.startswith("ud_"):
            tls_variants += ["ib", "\\" + tls_variant]
            break

    tls_variants = _powerset(tls_variants, with_empty_set=False)
    test_funcs = [test_ucx_tls_positive, test_ucx_tls_negative]
    for (tls_variant, test_func) in \
        itertools.product(tls_variants, test_funcs):
        test_func(",".join(tls_variant))

    # Test auxiliary transport negation
    test_cases_negative = [
        ("ib", {"ud_mlx5", "ud_verbs"}),
        ("ud,ud:aux", {"ud_mlx5", "ud_verbs"}),
        ("ud_v,ud_v:aux", {"ud_verbs"}),
        ("ud_x,ud_x:aux", {"ud_mlx5"}),
        ("ud_verbs,ud_verbs:aux", {"ud_verbs"}),
        ("ud_mlx5,ud_mlx5:aux", {"ud_mlx5"})
    ]

    for tls, forbidden_tls in test_cases_negative:
        test_ucx_tls_negative(tls, protocol="keepalive", forbidden_tls=forbidden_tls)

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
    print(f"directory \"{bin_prefix}\" does not exist")
    parser.print_help()
    exit(1)

ucx_info = bin_prefix + "/ucx_info"
ucx_info_args = " -e -u t -n "
ucx_info_eh_args = " -e -u et -n "

status, output = exec_cmd(ucx_info + " -c | grep -e \"UCX_RC_.*_MAX_NUM_EPS\"")
match = re.findall(r'\S+=(\d+)', output)
if match:
    rc_max_num_eps = int(max(match))
else:
    rc_max_num_eps = 0

status, output = exec_cmd("ibv_devinfo -l | tail -n+2 | head -n-1 | sed -e 's/^[ \t]*//' | grep -v '^smi[0-9]*$'")
dev_list = output.splitlines()
port = "1"

for dev in sorted(dev_list):
    status, dev_attrs = exec_cmd("ibv_devinfo -d " + dev + " -i " + port)
    if dev_attrs.find("PORT_ACTIVE") == -1:
        continue

    if not os.path.exists(f"/sys/class/infiniband/{dev}/ports/{port}/gids/0"):
        print("Skipping dummy device: ", dev)
        continue

    driver_name = os.path.basename(os.readlink(f"/sys/class/infiniband/{dev}/device/driver"))
    dev_name    = driver_name.split("_")[0] # should be mlx4 or mlx5
    if not dev_name in ['mlx4', 'mlx5']:
        print("Skipping unknown device: ", dev_name)
        continue

    if dev_attrs.find("Ethernet") == -1:
        dev_tl_map = am_tls[dev_name]
        dev_tl_override_map = am_tls[dev_name + "_override"]
        override = 1
    else:
        fw_ver = open(f"/sys/class/infiniband/{dev}/fw_ver").read()
        if parse_version(fw_ver) >= parse_version("16.23.0"):
            dev_tl_map = am_tls[dev_name+"_roce_dc"]
        else:
            dev_tl_map = am_tls[dev_name+"_roce_no_dc"]
        override = 0

    for n_eps in sorted(dev_tl_map):
        tl = find_am_transport(dev + ':' + port, n_eps)
        print(f"{dev}:{port} eps: {n_eps} expected am tl: {dev_tl_map[n_eps]} selected: {tl}")

        if dev_tl_map[n_eps] != tl:
            sys.exit(1)

        if override:
            tl = find_am_transport(f"{dev}:{port}", n_eps, 1)
            print(f"{dev}:{port} UCX_NUM_EPS=2 eps: {n_eps} expected am tl: \
                  {dev_tl_override_map[n_eps]} selected: {tl}")

            if dev_tl_override_map[n_eps] != tl:
                sys.exit(1)

        if n_eps >= (rc_max_num_eps * 2):
            test_fallback_from_rc(f"{dev}:{port}", n_eps)

# Test UCX_TLS configuration (TL choice according to "allow" and "negate" lists)
test_tls_allow_list(ucx_info)

sys.exit(0)

