#!/usr/bin/env python3
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Compare ucx_perftest device-CUDA results between a base build and a head
# build and flag bandwidth/latency regressions. Tolerates noisy shared CI
# nodes by taking the median across repeated runs and using a loose threshold.
#
# Input files are raw stdout of `ucx_perftest -b <batch>` (one file per run).
# A batch test prints a "+<name>----" header line followed by a "Final:" line
# with the result columns (see src/tools/perf/perftest_run.c).

import argparse
import os
import re
import statistics
import sys

HEADER_RE = re.compile(r"^\+([A-Za-z0-9_]+)")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


def read_test_names(config_path):
    names = []
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.split()[0])
    return names


def _metric_from_floats(name, floats):
    """Return (bandwidth_MBs, latency_us) for one result row.

    Multi-thread final row: iters, lat(us), bw(MB/s), msgrate        (4 cols)
    Single-thread final row: iters, lat_pctl, lat_mom, lat_total,
                             bw_mom, bw_total, mr_mom, mr_total       (8 cols)
    """
    if len(floats) >= 8:
        return floats[5], floats[3]
    if len(floats) >= 4:
        return floats[2], floats[1]
    return None, None


def parse_file(path, names):
    """Return {test_name: value} using the metric implied by the name.

    Each batch test prints a "+<name>----" header then a "Final:" result line.
    """
    name_set = set(names)
    results = {}
    current = None
    if not os.path.isfile(path):
        return results
    with open(path) as f:
        for line in f:
            m = HEADER_RE.match(line)
            if m and m.group(1) in name_set:
                current = m.group(1)
            elif current is not None and line.startswith("Final:"):
                floats = [float(x) for x in FLOAT_RE.findall(line)]
                bw, lat = _metric_from_floats(current, floats)
                val = bw if "_bw_" in current else lat
                if val is not None:
                    results[current] = val
                current = None
    return results


def median_by_test(paths, names):
    samples = {}
    for p in paths:
        for name, val in parse_file(p, names).items():
            samples.setdefault(name, []).append(val)
    return {name: statistics.median(vals) for name, vals in samples.items()}


def regression_pct(name, base, head):
    """Positive value == head is worse than base."""
    if base == 0:
        return 0.0
    if "_bw_" in name:          # bandwidth: higher is better
        return (base - head) / base * 100.0
    return (head - base) / base * 100.0   # latency: lower is better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True,
                    help="batch config file (test_types_ucp_device_cuda)")
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="max tolerated regression %% (default 15)")
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--head", nargs="+", required=True)
    args = ap.parse_args()

    names = read_test_names(args.names)
    base = median_by_test(args.base, names)
    head = median_by_test(args.head, names)

    if not base or not head:
        print("ERROR: no parseable result files (base=%d, head=%d). "
              "Did the perftest runs produce output?" % (len(base), len(head)),
              file=sys.stderr)
        return 1

    regressed = []
    print("%-34s %12s %12s %9s" % ("test", "base", "head", "regr%"))
    for name in names:
        b = base.get(name)
        h = head.get(name)
        if b is not None and h is None:
            # Ran on base but produced no result on head - a hang/crash is the
            # loudest regression, so fail rather than silently skip.
            print("%-34s %12.2f %12s   MISSING ON HEAD <== REGRESSION" %
                  (name, b, "-"))
            regressed.append((name, float("inf")))
            continue
        if b is None or h is None:
            print("%-34s %12s %12s   (no baseline)" %
                  (name, "-" if b is None else b, "-" if h is None else h))
            continue
        pct = regression_pct(name, b, h)
        flag = " <== REGRESSION" if pct > args.threshold else ""
        print("%-34s %12.2f %12.2f %8.1f%%%s" % (name, b, h, pct, flag))
        if pct > args.threshold:
            regressed.append((name, pct))

    if regressed:
        msg = "device perftest regression > %.0f%%: %s" % (
            args.threshold,
            ", ".join("%s %.0f%%" % (n, p) for n, p in regressed))
        # Loud, build-failing error (Azure annotation + non-zero exit).
        print("##vso[task.logissue type=error]" + msg)
        print("FAIL: " + msg, file=sys.stderr)
        return 1

    print("No device perftest regression above %.0f%% threshold." %
          args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
