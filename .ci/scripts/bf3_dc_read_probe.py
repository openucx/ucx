#!/usr/bin/env python3
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Probe DC RDMA READ completion on a BlueField-3 Blossom host.
#
#   raw                record host facts, measure the RC READ rate per RoCE
#                      GID and run raw ib_read_bw DC/RC cases with counters
#   screen <ucx-dir>   screen UCX DC settings on a proto-v2 ucx_perftest
#                      reproducer at the recv_align full message size
#
# Each run ends as PASS, ERROR (a side exited non-zero), HANG (no exit before
# the deadline) or SETUP (the run could not start, or the pod ran out of
# memory). Case results are data: only setup failures exit non-zero.
#

import ipaddress
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

DEV = "mlx5_0"
IB_PORT = 1
PORT_PATH = Path("/sys/class/infiniband/%s/ports/%d" % (DEV, IB_PORT))
CALICO_NET = ipaddress.ip_network("10.244.0.0/16")
TCP_PORT = 18600
GIB = 1 << 30
MIB = 1 << 20
# Used when the RC rate on the calico GID could not be measured
DEFAULT_RATE = 40.0
WORKDIR = Path(os.environ.get("WORKSPACE", ".")) / "bf3_probe"
RATE_FILE = WORKDIR / "calico_rc_rate"
CPUS = sorted(os.sched_getaffinity(0))
# Two CPUs per instance, as each CI pod gets; reuse the first pair if needed
CPU_PAIRS = [",".join(map(str, CPUS[i:i + 2])) for i in (0, 2)
             if len(CPUS) >= i + 2] or [",".join(map(str, CPUS))]
CPU_PAIRS += CPU_PAIRS[:1] * (2 - len(CPU_PAIRS))
TRANSPORT_ERROR = re.compile(r"retry count exceeded|retry exceeded|"
                             r"Endpoint timeout|syndrom|Failed status|"
                             r"Remote (access|operation) error|"
                             r"Local protection", re.IGNORECASE)
GENERIC_ERROR = re.compile(r"error|abort|fatal", re.IGNORECASE)


def log(*args):
    print(*args, flush=True)


def read_text(path):
    with suppress(OSError):
        return path.read_text(errors="replace").strip()
    return None


def shell(command, timeout=60, env=None):
    log("+", command)
    try:
        result = subprocess.run(command, shell=True, text=True,
                                errors="replace", env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        log(result.stdout.rstrip())
        return result.returncode
    except subprocess.TimeoutExpired as error:
        log("TIMEOUT", command, error.output)
        return None


def port_counters():
    snapshot = {}
    for subdir in ("hw_counters", "counters"):
        directory = PORT_PATH / subdir
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            with suppress(OSError, ValueError):
                snapshot[subdir + "/" + path.name] = int(path.read_text())
    return snapshot


def counter_delta(before, after):
    return {name: after[name] - before.get(name, 0) for name in sorted(after)
            if after[name] != before.get(name, 0)}


def memory_limit():
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        value = read_text(Path(path))
        if value and value.isdigit():
            return int(value)
    return None


def oom_kills():
    for path in ("/sys/fs/cgroup/memory.events",
                 "/sys/fs/cgroup/memory/memory.oom_control"):
        text = read_text(Path(path)) or ""
        match = re.search(r"oom_kill (\d+)", text)
        if match:
            return int(match.group(1))
    return 0


def gid_table():
    table = []
    for path in sorted((PORT_PATH / "gids").iterdir(),
                       key=lambda p: int(p.name)):
        gid = read_text(path)
        if gid is None or set(gid.replace(":", "")) == {"0"}:
            continue
        with suppress(ValueError):
            address = ipaddress.ip_address(gid)
            table.append({
                "index": int(path.name),
                "gid": gid,
                "type": read_text(PORT_PATH / "gid_attrs/types" / path.name),
                "ndev": read_text(PORT_PATH / "gid_attrs/ndevs" / path.name),
                "ipv4": address.ipv4_mapped,
            })
    return table


def roce_v2_ipv4(table):
    return [entry for entry in table
            if entry["ipv4"] and "v2" in (entry["type"] or "")]


def memory_info():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        info[name] = int(value.split()[0]) * 1024
    return info


def listening(port):
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in (read_text(Path(table)) or "").splitlines()[1:]:
            fields = line.split()
            if fields[1].endswith(":%04X" % port) and fields[3] == "0A":
                return True
    return False


def start(name, command, env, prefix):
    path = WORKDIR / ("%s-%s.log" % (prefix, name))
    with open(path, "w") as output:
        proc = subprocess.Popen(command, env=env, stdout=output,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
    return {"name": name, "proc": proc, "path": path}


def backtrace(label, peer):
    if not shutil.which("gdb"):
        log("GDB unavailable", label, peer["name"])
        return
    try:
        trace = subprocess.run(
            ["timeout", "--kill-after=5s", "30s", "gdb", "-nx", "-batch",
             "-iex", "set debuginfod enabled off",
             "-ex", "set pagination off", "-ex", "thread apply all bt",
             "-p", str(peer["proc"].pid)],
            text=True, errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=40)
        log("BACKTRACE", label, peer["name"], trace.stdout)
    except (OSError, subprocess.SubprocessError) as error:
        log("BACKTRACE FAILED", label, peer["name"], error)


def kill(peer):
    proc = peer["proc"]
    if proc.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        log("WEDGED", peer["name"], "pid", proc.pid, "wchan",
            read_text(Path("/proc/%d/wchan" % proc.pid)))
        raise SystemExit("a killed process did not exit; device may be wedged")


def run_pairs(label, pairs, timeout, env=None):
    """Run (port, server, client) pairs concurrently and classify each one."""
    env = env or os.environ
    WORKDIR.mkdir(parents=True, exist_ok=True)
    before, oom_before = port_counters(), oom_kills()
    started = time.monotonic()
    deadline = started + timeout
    states = []
    for port, server, client in pairs:
        log("CASE", label, "port", port, shlex.join(client))
        state = {"port": port, "peers": [], "status": None, "detail": ""}
        states.append(state)
        if listening(port):
            state.update(status="SETUP", detail="port %d busy" % port,
                         end=time.monotonic())
            continue
        state["peers"].append(start("server", server, env,
                                    "%s-%d" % (label, port)))
    for (port, _, client), state in zip(pairs, states):
        if state["status"]:
            continue
        server = state["peers"][0]["proc"]
        ready = time.monotonic() + 30
        while (server.poll() is None and time.monotonic() < ready and
               not listening(port)):
            time.sleep(0.1)
        if not listening(port):
            state.update(status="SETUP", end=time.monotonic(),
                         detail="server did not listen, rc %s" % server.poll())
            continue
        state["peers"].append(start("client", client, env,
                                    "%s-%d" % (label, port)))
    while True:
        now = time.monotonic()
        pending = False
        for state in states:
            if state["status"]:
                continue
            rcs = [peer["proc"].poll() for peer in state["peers"]]
            failed = [(peer["name"], rc)
                      for peer, rc in zip(state["peers"], rcs)
                      if rc not in (None, 0)]
            if failed:
                # The other side gets 10 s to notice before it is killed
                state.setdefault("failed_at", now)
                state.setdefault("first_failed", failed[0])
                if (None not in rcs or now - state["failed_at"] >= 10 or
                        now >= deadline):
                    state.update(status="ERROR", end=now,
                                 detail="%s rc %s first" %
                                 state["first_failed"])
                else:
                    pending = True
            elif None not in rcs:
                state.update(status="PASS", end=now)
            elif now >= deadline:
                state.update(status="HANG", end=now,
                             detail="no exit within %ds" % timeout)
                for peer in state["peers"]:
                    if peer["proc"].poll() is None:
                        log("WAIT CHANNEL", label, peer["name"], read_text(
                            Path("/proc/%d/wchan" % peer["proc"].pid)))
                        backtrace(label, peer)
            else:
                pending = True
        if not pending:
            break
        time.sleep(0.2)
    for state in states:
        for peer in state["peers"]:
            kill(peer)
    delta = counter_delta(before, port_counters())
    oom = oom_kills() - oom_before
    results = []
    for state in states:
        outputs = {}
        for peer in state["peers"]:
            outputs[peer["name"]] = read_text(peer["path"]) or ""
            log("PEER", label, "port", state["port"], peer["name"], "RC",
                peer["proc"].returncode)
            log(outputs[peer["name"]])
        text = "\n".join(outputs.values())
        signature = None
        if state["status"] != "PASS":
            signature = (TRANSPORT_ERROR.search(text) or
                         GENERIC_ERROR.search(text))
        signature_line = ""
        if signature:
            start_index = text.rfind("\n", 0, signature.start()) + 1
            end_index = text.find("\n", signature.end())
            signature_line = text[start_index:end_index if end_index >= 0
                                  else None].strip()[:240]
        if oom and state["status"] in ("ERROR", "HANG"):
            state.update(status="SETUP", detail="pod OOM kill")
        elapsed = state["end"] - started
        log("RESULT", label, "port", state["port"], state["status"],
            "elapsed=%.1fs" % elapsed, state["detail"], signature_line)
        results.append({"status": state["status"], "elapsed": elapsed,
                        "outputs": outputs, "signature": signature_line,
                        "transport_error": bool(TRANSPORT_ERROR.search(text))})
    log("COUNTERS", label, delta, "oom_kill +%d" % oom if oom else "")
    return results


def rate_timeout(nbytes, rate, instances=1):
    expected = nbytes * instances / (rate * MIB)
    return max(180, 3 * expected + 60)


def read_bw(label, gid, conn, size, iters, timeout, extra=()):
    command = ["taskset", "-c", CPU_PAIRS[0], "stdbuf", "-oL", "ib_read_bw",
               "-d", DEV, "-i", str(IB_PORT), "-x", str(gid), "-c", conn,
               "-q", "1", "-s", str(size), "-n", str(iters), "-Q", "1",
               "-u", "18", "--tclass", "106",
               "-p", str(TCP_PORT)] + list(extra)
    result = run_pairs(label, [(TCP_PORT, command, command + ["127.0.0.1"])],
                       timeout)[0]
    result["rate"] = None
    client = result["outputs"].get("client", "")
    for line in client.splitlines():
        match = re.match(r"^\s*%d\s+%d\s+[\d.]+\s+([\d.]+)\s" % (size, iters),
                         line)
        if match:
            result["rate"] = float(match.group(1))
    result["banner"] = sorted(set(
        re.sub(r"\s+", " ", line.strip()) for line in client.splitlines()
        if re.search(r"TX depth|Outstand reads|GID index|Enhanced Reorder|"
                     r"Connection type|Mtu", line)))
    log("RATE", label, result["rate"], "MB/s")
    return result


def raw():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    for command in ("uname -a", "hostname", "nproc",
                    "grep -E '^(MemTotal|MemFree|MemAvailable):' /proc/meminfo",
                    "cat /sys/fs/cgroup/memory.max 2>/dev/null || true",
                    "ibv_devinfo -d mlx5_0 | grep -E 'fw_ver|board_id|state|active_mtu|link_layer'",
                    "ibv_devinfo -d mlx5_1 | grep -E 'fw_ver|state|active_mtu|link_layer' || true",
                    "ip -br addr || true",
                    "command -v ib_read_bw gdb || true"):
        shell(command)
    log("CPU pairs:", CPU_PAIRS)

    table = gid_table()
    for entry in table:
        log("GID", entry["index"], entry["gid"], entry["type"], entry["ndev"],
            entry["ipv4"])
    candidates = roce_v2_ipv4(table)
    if candidates:
        log("UCX default GID (first RoCE v2 IPv4):", candidates[0]["index"],
            candidates[0]["ipv4"])
    calico = [entry for entry in candidates if entry["ipv4"] in CALICO_NET]
    if not calico:
        raise SystemExit("SETUP: no RoCE v2 GID in %s" % CALICO_NET)
    calico_gid = calico[0]["index"]
    if calico_gid != candidates[0]["index"]:
        log("WARNING: UCX picks GID %d, not the calico GID %d" %
            (candidates[0]["index"], calico_gid))

    help_text = subprocess.run(["ib_read_bw", "--help"], text=True,
                               errors="replace", stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT).stdout
    if "--tclass" not in help_text:
        raise SystemExit("SETUP: ib_read_bw cannot set the traffic class")
    no_reorder = (["--no_enhanced_reorder"]
                  if "--no_enhanced_reorder" in help_text else [])
    log("Enhanced-reorder switch:", no_reorder or "unavailable")

    # (a) RC READ rate on every RoCE v2 IPv4 GID, at the default MTU and 512
    rates = {}
    for entry in candidates:
        for mtu in (None, 512):
            extra = no_reorder + (["-m", str(mtu)] if mtu else [])
            label = "RATE-RC-gid%d-mtu%s" % (entry["index"], mtu or "default")
            result = read_bw(label, entry["index"], "RC", 8 * MIB, 50, 120,
                             extra)
            if result["status"] == "SETUP":
                raise SystemExit("SETUP: %s could not start" % label)
            if mtu is None:
                rates[entry["index"]] = result["rate"]
    log("RC rates by GID (MB/s):", rates)
    calico_rate = rates.get(calico_gid)
    RATE_FILE.write_text(str(calico_rate or DEFAULT_RATE))

    # (b) Raw DC and RC cases on the calico GID, the path UCX takes
    cases = [
        ("C0-DC-1G-n5-t2-o2", calico_gid, "DC", GIB, 5, 2, 2),
        ("C1-DC-1G-n5-t3-o3", calico_gid, "DC", GIB, 5, 3, 3),
        ("C2-DC-512M-n5-t4-o4", calico_gid, "DC", 512 * MIB, 5, 4, 4),
        ("C3-RC-1G-n5-t2-o2", calico_gid, "RC", GIB, 5, 2, 2),
    ]
    fast = [index for index, rate in rates.items()
            if index != calico_gid and rate and calico_rate and
            rate >= 10 * calico_rate]
    if fast:
        cases.append(("C4-DC-1G-n5-t2-o2-gid%d" % fast[0], fast[0], "DC",
                      GIB, 5, 2, 2))
    else:
        log("C4 skipped: no GID is 10x faster than the calico GID")
    summary = []
    for label, gid, conn, size, iters, depth, reads in cases:
        extra = no_reorder + ["-t", str(depth), "-o", str(reads)]
        timeout = rate_timeout(size * iters,
                               rates.get(gid) or calico_rate or DEFAULT_RATE)
        result = read_bw(label, gid, conn, size, iters, timeout, extra)
        text = "\n".join(result["outputs"].values())
        failure = re.findall(r"(Failed status \d+.*|scnt=\d+, ccnt=\d+)", text)
        summary.append((label, result["status"], "%.1fs" % result["elapsed"],
                        failure[:3], result["banner"]))
    for line in summary:
        log("SUMMARY", *line)
    if any(line[1] == "SETUP" for line in summary):
        raise SystemExit("SETUP: a raw case could not run")


def screen(ucx_dir):
    WORKDIR.mkdir(parents=True, exist_ok=True)
    perftest = os.path.join(ucx_dir, "bin", "ucx_perftest")
    ucx_info = os.path.join(ucx_dir, "bin", "ucx_info")
    rate = float(read_text(RATE_FILE) or DEFAULT_RATE)
    memory = memory_info()
    limit = memory_limit()
    full_size = min(2500 * MIB, memory["MemTotal"] // 16)
    # Every ucx_perftest process pins a send and a receive buffer
    pair_size = full_size
    if limit:
        pair_size = min(full_size, (limit - 2 * GIB) // 8)
    log("MemTotal/16:", memory["MemTotal"] // 16, "MemFree/4:",
        memory["MemFree"] // 4, "full size:", full_size, "cgroup limit:",
        limit, "2-instance size:", pair_size, "calico RC rate:", rate)

    inherited = sorted(name for name in os.environ if name.startswith("UCX_"))
    if inherited:
        log("Dropping inherited", inherited)
    base_env = {name: value for name, value in os.environ.items()
                if not name.startswith("UCX_")}
    base_env.update(UCX_TLS="dc_x", UCX_NET_DEVICES="%s:%d" % (DEV, IB_PORT),
                    UCX_HANDLE_ERRORS="bt", UCX_RNDV_SCHEME="get_zcopy")
    shell("%s -v" % ucx_info, env=base_env)
    shell("%s -d -t dc_mlx5 | grep -iE 'transport|device|get_zcopy|bandwidth'"
          % ucx_info, env=base_env)
    shell("%s -c -f | grep -E '^UCX_(DC_MLX5_(MAX_RD_ATOMIC|TX_NUM_GET_BYTES|"
          "DCI_FULL_HANDSHAKE|DCT_FULL_HANDSHAKE|MAX_GET_ZCOPY|ROCE_SUBNETS)|"
          "IB_ROCE_SUBNETS|PROTO_ENABLE)='" % ucx_info, env=base_env)

    results = {}

    def run_variant(name, env_extra, test, size, pairs=1, proto_info=False):
        env = dict(base_env, **env_extra)
        if proto_info:
            env["UCX_PROTO_INFO"] = "y"
        assert "UCX_PROTO_ENABLE" not in env
        label = "%s-%s-%dB%s" % (name, test, size,
                                 "-x%d" % pairs if pairs > 1 else "")
        log("VARIANT", label, env_extra)
        pair_commands = []
        for i in range(pairs):
            port = TCP_PORT + i
            prefix = ["taskset", "-c", CPU_PAIRS[i], "stdbuf", "-oL",
                      perftest, "-p", str(port)]
            pair_commands.append((port, prefix, prefix + [
                "127.0.0.1", "-t", test, "-s", str(size), "-n", "1",
                "-w", "0"]))
        timeout = (60 if size < GIB else
                   rate_timeout(size, rate, instances=pairs))
        runs = run_pairs(label, pair_commands, timeout, env)
        key = (name, test, size, pairs)
        results.setdefault(key, []).extend(run["status"] for run in runs)
        return runs

    def reproduced(run):
        return (run["status"] == "HANG" or
                (run["status"] == "ERROR" and run["transport_error"]))

    # The reproducers must work with the default settings at a small size
    for test in ("ucp_get", "tag_bw"):
        if run_variant("SANITY", {}, test, 64 * MIB)[0]["status"] != "PASS":
            raise SystemExit("SETUP: %s fails at 64 MiB with defaults" % test)

    # V0 picks the reproducer: proto-v2 RMA GET first, then tag rendezvous GET
    reproducer = None
    for test in ("ucp_get", "tag_bw"):
        for attempt in range(2):
            run = run_variant("V0", {}, test, full_size,
                              proto_info=(attempt == 0))[0]
            if run["status"] == "SETUP":
                raise SystemExit("SETUP: V0 %s could not run" % test)
            if reproduced(run):
                reproducer = test
                break
        if reproducer:
            break
    if reproducer is None:
        log("DECISION: V0 did not reproduce with ucp_get or tag_bw at %d B; "
            "no screen on this host" % full_size)
        return

    variants = [
        ("V1", {"UCX_DC_MLX5_MAX_RD_ATOMIC": "1"}),
        ("V2", {"UCX_DC_MLX5_TX_NUM_GET_BYTES": "1g"}),
        ("V3", {"UCX_DC_MLX5_DCI_FULL_HANDSHAKE": "yes",
                "UCX_DC_MLX5_DCT_FULL_HANDSHAKE": "yes"}),
    ]
    if [entry for entry in roce_v2_ipv4(gid_table())
            if entry["ipv4"] not in CALICO_NET]:
        variants.append(("V4", {"UCX_IB_ROCE_SUBNETS": "^%s" % CALICO_NET}))
    else:
        log("V4 skipped: no RoCE v2 IPv4 GID outside %s" % CALICO_NET)

    # SANITY passed with the defaults at 64 MiB, so a variant that fails there
    # fails because of its setting: report it and skip its full-size runs
    usable = []
    for name, env_extra in variants:
        status = run_variant(name, env_extra, reproducer, 64 * MIB)[0]["status"]
        if status == "PASS":
            usable.append((name, env_extra))
        else:
            log("VARIANT FAILS", name, status, "at 64 MiB; not screened")

    for round_index in range(3):
        for name, env_extra in usable:
            run_variant(name, env_extra, reproducer, full_size)
        if round_index == 1:
            run_variant("V0", {}, reproducer, full_size)

    preference = ["V4", "V3", "V1", "V2"]
    clean = sorted((variant for variant in usable
                    if set(results[(variant[0], reproducer, full_size, 1)]) ==
                    {"PASS"}),
                   key=lambda variant: preference.index(variant[0]))
    if clean:
        name, env_extra = clean[0]
        log("CONCURRENCY candidate", name, env_extra, "size", pair_size)
        run_variant(name, env_extra, reproducer, pair_size, pairs=2)
    else:
        log("CONCURRENCY skipped: no variant passed all its runs")

    for (name, test, size, pairs), statuses in sorted(results.items()):
        counts = {status: statuses.count(status)
                  for status in ("PASS", "ERROR", "HANG", "SETUP")
                  if status in statuses}
        log("SUMMARY", name, test, "%dB" % size, "instances=%d" % pairs,
            counts)
    if any("SETUP" in statuses for statuses in results.values()):
        raise SystemExit("SETUP: a UCX run could not run")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "raw":
        raw()
    elif len(sys.argv) == 3 and sys.argv[1] == "screen":
        screen(sys.argv[2])
    else:
        raise SystemExit("usage: %s raw | screen <ucx-dir>" % sys.argv[0])


if __name__ == "__main__":
    main()
