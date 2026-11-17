# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from time import sleep, time

import pytest
import yaml

import dask
from distributed import Client
from distributed.utils import get_ip, open_port
from distributed.utils_test import popen

from distributed_ucxx.ucxx import _prepare_ucx_config
from distributed_ucxx.utils_test import gen_test

try:
    HOST = get_ip()
except Exception:
    HOST = "127.0.0.1"

rmm = pytest.importorskip("rmm")


@gen_test()
async def test_ucx_config(ucxx_loop, cleanup):
    ucx = {
        "nvlink": True,
        "infiniband": True,
        "rdmacm": False,
        "tcp": True,
        "cuda-copy": True,
    }

    with dask.config.set({"distributed-ucxx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {
            "TLS": "rc,tcp,cuda_copy,cuda_ipc",
            "SOCKADDR_TLS_PRIORITY": "tcp",
        }
        assert ucx_environment == {}

    ucx = {
        "nvlink": False,
        "infiniband": True,
        "rdmacm": False,
        "tcp": True,
        "cuda-copy": False,
    }

    with dask.config.set({"distributed-ucxx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {"TLS": "rc,tcp", "SOCKADDR_TLS_PRIORITY": "tcp"}
        assert ucx_environment == {}

    ucx = {
        "nvlink": False,
        "infiniband": True,
        "rdmacm": True,
        "tcp": True,
        "cuda-copy": True,
    }

    with dask.config.set({"distributed-ucxx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {
            "TLS": "rc,tcp,cuda_copy",
            "SOCKADDR_TLS_PRIORITY": "rdmacm",
        }
        assert ucx_environment == {}

    ucx = {
        "nvlink": None,
        "infiniband": None,
        "rdmacm": None,
        "tcp": None,
        "cuda-copy": None,
    }

    with dask.config.set({"distributed-ucxx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {}
        assert ucx_environment == {}

    ucx = {
        "nvlink": False,
        "infiniband": True,
        "rdmacm": True,
        "tcp": True,
        "cuda-copy": True,
    }

    ucx["environment"] = {
        "tls": "all",
        "memtrack-dest": "stdout",
    }

    with dask.config.set({"distributed-ucxx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {
            "TLS": "rc,tcp,cuda_copy",
            "SOCKADDR_TLS_PRIORITY": "rdmacm",
        }
        assert ucx_environment == {"UCX_MEMTRACK_DEST": "stdout"}


@contextmanager
def start_dask_scheduler(
    env: list[str], protocol: str, max_attempts: int = 5, timeout: int = 10
):
    """
    Start Dask scheduler in subprocess.

    Attempts to start a Dask scheduler in subprocess, if the port is not available
    retry on a different port up to a maximum of `max_attempts` attempts. The stdout
    and stderr of the process is read to determine whether the scheduler failed to
    bind to port or succeeded, and ensures no more than `timeout` seconds are awaited
    for between reads.

    Parameters
    ----------
    env: list[str]
        Environment variables to start scheduler process with.
    protocol: str
        Communication protocol name to use, either "ucx" or "ucxx".
    max_attempts: int
        Maximum attempts to try to open scheduler.
    timeout: int
        Time to wait while reading stdout/stderr of subprocess.
    """
    port = open_port()
    for _ in range(max_attempts):
        with popen(
            [
                "dask",
                "scheduler",
                "--no-dashboard",
                "--protocol",
                protocol,
                "--port",
                str(port),
            ],
            env=env,
            capture_output=True,  # Capture stdout and stderr
        ) as scheduler_process:
            # Check if the scheduler process started successfully by streaming output
            try:
                start_time = time()
                while True:
                    if time() - start_time > timeout:
                        raise TimeoutError("Timeout while waiting for scheduler output")

                    # Use select to wait for data with a timeout
                    line = scheduler_process.stdout.readline()
                    if not line:
                        break  # End of output
                    print(
                        line.decode(), end=""
                    )  # Since capture_output=True, print the line here
                    if b"Scheduler at:" in line:
                        # Scheduler is now listening
                        break
                    elif b"UCXXBusyError" in line:
                        raise Exception("UCXXBusyError detected in scheduler output")
            except Exception:
                port += 1
            else:
                yield scheduler_process, port
                return
    else:
        pytest.fail(f"Failed to start dask scheduler after {max_attempts} attempts.")


@pytest.mark.parametrize("protocol", ["ucx", "ucxx"])
@pytest.mark.skipif(
    int(os.environ.get("UCXPY_ENABLE_PYTHON_FUTURE", "0")) != 0,
    reason="Workers running without a `Nanny` can't be closed properly",
)
def test_ucx_config_w_env_var(ucxx_loop, cleanup, loop, protocol):
    def current_device_resource_is_pool():
        import rmm

        return isinstance(
            rmm.mr.get_current_device_resource(), rmm.mr.PoolMemoryResource
        )

    env = os.environ.copy()
    env["DASK_DISTRIBUTED__RMM__POOL_SIZE"] = "1000.00 MB"

    with start_dask_scheduler(env=env, protocol=protocol) as scheduler_process_port:
        scheduler_process, scheduler_port = scheduler_process_port
        sched_addr = f"{protocol}://127.0.0.1:{scheduler_port}"
        print(f"{sched_addr=}", flush=True)

        with popen(
            [
                "dask",
                "worker",
                sched_addr,
                "--host",
                "127.0.0.1",
                "--no-dashboard",
                "--protocol",
                protocol,
                "--no-nanny",
            ],
            env=env,
        ):
            with Client(sched_addr, loop=loop, timeout=60) as c:
                while not c.scheduler_info()["workers"]:
                    sleep(0.1)

                # Check for RMM pool resource type
                rmm_resource = c.run_on_scheduler(current_device_resource_is_pool)
                assert rmm_resource

                rmm_resource_workers = c.run(current_device_resource_is_pool)
                assert all(rmm_resource_workers.values())


def test_schema():
    jsonschema = pytest.importorskip("jsonschema")

    root_dir = pathlib.Path(__file__).parent.parent
    config = yaml.safe_load((root_dir / "distributed-ucxx.yaml").read_text())
    schema = yaml.safe_load((root_dir / "distributed-ucxx-schema.yaml").read_text())

    jsonschema.validate(config, schema)
