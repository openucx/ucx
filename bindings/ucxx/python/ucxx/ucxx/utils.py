# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import fcntl
import glob
import logging
import multiprocessing as mp
import os
import socket
import struct
import time

import numpy as np

mp = mp.get_context("spawn")


try:
    from nvtx import annotate as nvtx_annotate
except ImportError:
    # If nvtx module is not installed, `annotate` yields only.
    from contextlib import contextmanager

    @contextmanager
    def nvtx_annotate(message=None, color=None, domain=None):
        yield


try:
    from dask.utils import format_bytes, format_time, parse_bytes
except ImportError:

    def format_time(x):
        if x < 1e-6:
            return f"{x * 1e9:.3f} ns"
        if x < 1e-3:
            return f"{x * 1e6:.3f} us"
        if x < 1:
            return f"{x * 1e3:.3f} ms"
        else:
            return f"{x:.3f} s"

    def format_bytes(x):
        """Return formatted string in B, KiB, MiB, GiB or TiB"""
        if x < 1024:
            return f"{x} B"
        elif x < 1024**2:
            return f"{x / 1024:.2f} KiB"
        elif x < 1024**3:
            return f"{x / 1024**2:.2f} MiB"
        elif x < 1024**4:
            return f"{x / 1024**3:.2f} GiB"
        else:
            return f"{x / 1024**4:.2f} TiB"

    parse_bytes = None


def print_separator(separator="-", length=80):
    """Print a single separator character multiple times"""
    print(separator * length)


def print_key_value(key, value, key_length=25):
    """Print a key and value with fixed key-field length"""
    print(f"{key: <{key_length}} | {value}")


def print_multi(values, key_length=25):
    """Print a key and value with fixed key-field length"""
    assert isinstance(values, tuple) or isinstance(values, list)
    assert len(values) > 1

    print_str = "".join(f"{s: <{key_length}} | " for s in values[:-1])
    print_str += values[-1]
    print(print_str)


def hmean(a):
    """Harmonic mean"""
    if len(a):
        return 1 / np.mean(1 / a)
    else:
        return 0


def get_ucxpy_logger():
    """
    Get UCX-Py logger with custom formatting

    Returns
    -------
    logger : logging.Logger
        Logger object

    Examples
    --------
    >>> logger = get_ucxpy_logger()
    >>> logger.warning("Test")
    [1585175070.2911468] [dgx12:1054] UCXPY  WARNING Test
    """

    _level_enum = logging.getLevelName(os.getenv("UCXPY_LOG_LEVEL", "WARNING"))
    logger = logging.getLogger("ucx")

    # Avoid duplicate logging
    logger.propagate = False

    class LoggingFilter(logging.Filter):
        def filter(self, record):
            record.hostname = socket.gethostname()
            record.timestamp = str("%.6f" % time.time())
            return True

    formatter = logging.Formatter(
        "[%(timestamp)s] [%(hostname)s:%(process)d] UCXPY  %(levelname)s %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(LoggingFilter())
    logger.addHandler(handler)

    logger.setLevel(_level_enum)

    return logger


def get_address(ifname=None, use_ipv6=False):
    """
    Get the address associated with a network interface.

    Parameters
    ----------
    ifname : str
        The network interface name to find the address for.
        If None, it uses the value of environment variable `UCXPY_IFNAME`
        and if `UCXPY_IFNAME` is not set it defaults to "ib0"
        An OSError is raised for invalid interfaces.
    use_ipv6 : bool
        Whether to get IPv6 addresses instead of the IPv4 default.
        NOTE: Requires the `psutil` package.

    Raises
    ------
    OSError
        If no device was found with the specified `ifname`, or no suitable
        devices were found when `ifname=None`.

    Returns
    -------
    address : str
        The inet addr associated with an interface.

    Examples
    --------
    >>> get_address()
    '10.33.225.160'

    >>> get_address(ifname='lo')
    '127.0.0.1'

    >>> get_address(ifname='lo', use_ipv6=True)
    '::1'
    """

    def _get_address(ifname):
        if use_ipv6:
            import psutil

            addrs = psutil.net_if_addrs()
            if ifname in addrs:
                for addr in addrs[ifname]:
                    if addr.family == socket.AF_INET6:
                        return addr.address.split("%")[0]
            raise OSError("No such device")
        else:
            ifname = ifname.encode()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                return socket.inet_ntoa(
                    fcntl.ioctl(
                        s.fileno(),
                        0x8915,  # SIOCGIFADDR
                        struct.pack("256s", ifname[:15]),
                    )[20:24]
                )

    def _try_interfaces():
        prefix_priority = ["ib", "eth", "en"]
        iftypes = {p: [] for p in prefix_priority}
        for i in glob.glob("/sys/class/net/*"):
            name = i.split("/")[-1]
            for p in prefix_priority:
                if name.startswith(p):
                    iftypes[p].append(name)
        for p in prefix_priority:
            iftype = iftypes[p]
            iftype.sort()
            for i in iftype:
                try:
                    return _get_address(i)
                except OSError:
                    pass
        raise OSError("No devices found")

    if ifname is None:
        ifname = os.environ.get("UCXPY_IFNAME")

    if ifname is not None:
        return _get_address(ifname)
    else:
        return _try_interfaces()
