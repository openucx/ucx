# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import functools
import gc
import inspect
import os
import weakref

import pytest
import pytest_asyncio

import ucxx
from ucxx._lib_async.pytest_stash_keys import ASYNCIO_PLUGIN_TIMEOUT_STASH_KEY


_CALL_REPORT_STASH_KEY = pytest.StashKey[pytest.TestReport]()

# Prevent calls such as `cudf = pytest.importorskip("cudf")` from initializing
# a CUDA context. Such calls may cause tests that must initialize the CUDA
# context on the appropriate device to fail.
# For example, without `RAPIDS_NO_INITIALIZE=True`, `test_benchmark_cluster`
# will succeed if running alone, but fails when all tests are run in batch.
os.environ["RAPIDS_NO_INITIALIZE"] = "True"


def pytest_runtest_teardown(item, nextitem):
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print(msg)


class CustomEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Custom event loop policy providing custom event loop with UCXX setup/teardown."""

    def new_event_loop(self):
        loop = super().new_event_loop()
        loop.set_exception_handler(handle_exception)
        return loop


@pytest.fixture(scope="session")
def event_loop_policy():
    """Provide a custom event loop policy for the entire test session."""
    policy = CustomEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)
    return policy


class _CreatedResources:
    """Weakly observe resources created by a test and their handler futures."""

    def __init__(self):
        self._resources = []
        self._handler_trackers = set()

    def add(self, resource):
        self._resources.append((type(resource).__name__, weakref.ref(resource)))
        handler_tracker = getattr(resource, "_handler_tracker", None)
        if handler_tracker is not None:
            self._handler_trackers.add(handler_tracker)
        return resource

    def _progress(self):
        ctx = ucxx.core._ctx
        return (
            ctx.worker.progress
            if ctx is not None and not ctx.progress_mode.startswith("thread")
            else None
        )

    async def _wait_for_handlers(self, timeout, progress):
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(
                        tracker.wait(progress=progress)
                        for tracker in self._handler_trackers
                    ),
                    return_exceptions=True,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            active = sum(tracker.active_count for tracker in self._handler_trackers)
            raise AssertionError(
                "UCXX listener handlers did not finish within "
                f"{timeout} seconds ({active} still active)"
            ) from e

    async def wait_for_release(self, timeout=5.0):
        """Wait for handlers and implicitly-owned resources to be released."""
        progress = self._progress()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        await self._wait_for_handlers(timeout, progress)
        while True:
            # Task, future, and CUDA completion callbacks may release their last
            # resource references after the test coroutine has returned.
            await asyncio.sleep(0)
            gc.collect()
            live = self.live_context_owners
            if not live:
                return
            if loop.time() >= deadline:
                raise AssertionError(
                    "UCXX resources retained their context after the test "
                    f"returned: {live}"
                )
            if progress is not None:
                progress()
            await asyncio.sleep(0.01)

    async def close(self, timeout=5.0):
        """Explicitly close resources retained by a failed test traceback."""
        for _, ref in reversed(self._resources):
            resource = ref()
            if resource is None:
                continue
            abort = getattr(resource, "abort", None)
            if abort is not None:
                abort()
            elif not resource.closed:
                resource.close()

        await self._wait_for_handlers(timeout, self._progress())

    @property
    def live_context_owners(self):
        live = []
        for kind, ref in self._resources:
            resource = ref()
            if resource is not None and getattr(resource, "_ctx", None) is not None:
                live.append(kind)
        return live


@pytest_asyncio.fixture(autouse=True)
async def ucxx_setup_teardown(monkeypatch, request):
    """Setup UCXX and verify implicit resource cleanup before closing the loop."""
    resources = _CreatedResources()

    create_listener = ucxx.core.create_listener
    create_endpoint = ucxx.core.create_endpoint
    create_endpoint_from_worker_address = ucxx.core.create_endpoint_from_worker_address

    @functools.wraps(create_listener)
    def tracked_create_listener(*args, **kwargs):
        return resources.add(create_listener(*args, **kwargs))

    @functools.wraps(create_endpoint)
    async def tracked_create_endpoint(*args, **kwargs):
        return resources.add(await create_endpoint(*args, **kwargs))

    @functools.wraps(create_endpoint_from_worker_address)
    async def tracked_create_endpoint_from_worker_address(*args, **kwargs):
        return resources.add(await create_endpoint_from_worker_address(*args, **kwargs))

    monkeypatch.setattr(ucxx, "create_listener", tracked_create_listener)
    monkeypatch.setattr(ucxx.core, "create_listener", tracked_create_listener)
    monkeypatch.setattr(ucxx, "create_endpoint", tracked_create_endpoint)
    monkeypatch.setattr(ucxx.core, "create_endpoint", tracked_create_endpoint)
    monkeypatch.setattr(
        ucxx,
        "create_endpoint_from_worker_address",
        tracked_create_endpoint_from_worker_address,
    )
    monkeypatch.setattr(
        ucxx.core,
        "create_endpoint_from_worker_address",
        tracked_create_endpoint_from_worker_address,
    )

    ucxx.reset()
    yield
    call_report = request.node.stash.get(_CALL_REPORT_STASH_KEY, None)
    if call_report is not None and call_report.passed:
        await resources.wait_for_release()
    else:
        # Failed-test tracebacks retain local resources until after teardown.
        await resources.close()
    ucxx.reset()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Remember whether the test call passed for failure-safe UCXX teardown."""
    report = yield
    if report.when == "call":
        item.stash[_CALL_REPORT_STASH_KEY] = report
    return report


def _asyncio_plugin_timeout_seconds(item: pytest.Item) -> float:
    """Timeout used by asyncio test wrapper and exposed via ``config.stash``."""
    timeout_marker = item.get_closest_marker("asyncio_timeout")
    slow_marker = item.get_closest_marker("slow")
    default_timeout = 600.0 if slow_marker else 60.0
    return float(timeout_marker.args[0]) if timeout_marker else default_timeout


def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Record the effective asyncio timeout before the test body runs.

    ``compute_timeouts(pytestconfig)`` reads this from ``config.stash``. We use stash
    instead of ``config.cache`` because the cache backend is unreliable on
    ``pytest-xdist`` workers (values may not round-trip or may be isolated per hook).
    """
    if not isinstance(item, pytest.Function):
        return
    timeout = _asyncio_plugin_timeout_seconds(item)
    if timeout <= 0.0:
        raise ValueError("The `pytest.mark.asyncio_timeout` value must be positive.")
    item.config.stash[ASYNCIO_PLUGIN_TIMEOUT_STASH_KEY] = timeout


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    """
    Add timeout for tests, and optionally rerun on failure.

    Add timeout for tests with `pytest.mark.asyncio_timeout` marker as specified by the
    decorator, otherwise a default timeout of 60 seconds for regular tests and 600
    seconds for tests marked slow.

    Optionally rerun the test if it failed, for that the test has to be marked with
    `pytest.mark.rerun_on_failure(reruns)`. This is similar to `pytest-rerunfailures`,
    but that module closes the event loop before this function has awaited, making the
    two incompatible.

    The timeout value is made available to the test functions via `pytestconfig`. This
    can be used to determine internal timeouts, for example to ensure subprocesses
    timeout before the test timeout hits and thus prints internal information, such as
    the call stack. The timeout is stored in ``config.stash`` (see
    ``pytest_runtest_setup`` and ``compute_timeouts``).
    """
    timeout = _asyncio_plugin_timeout_seconds(pyfuncitem)
    pyfuncitem.config.stash[ASYNCIO_PLUGIN_TIMEOUT_STASH_KEY] = timeout
    rerun_marker = pyfuncitem.get_closest_marker("rerun_on_failure")

    if rerun_marker and len(rerun_marker.args) >= 0:
        reruns = rerun_marker.args[0]
        if not isinstance(reruns, int) or reruns < 0:
            raise ValueError("The `pytest.mark.rerun` value must be a positive integer")
    else:
        reruns = 1

    if inspect.iscoroutinefunction(pyfuncitem.obj) and timeout > 0.0:

        async def wrapped_obj(*args, **kwargs):
            for i in range(reruns):
                try:
                    try:
                        return await asyncio.wait_for(
                            inner_obj(*args, **kwargs), timeout=timeout
                        )
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pytest.fail(
                            f"{pyfuncitem.name} timed out after {timeout} seconds."
                        )
                except Exception as e:
                    if i == (reruns - 1):
                        raise e
                else:
                    break

        inner_obj = pyfuncitem.obj
        pyfuncitem.obj = wrapped_obj

    yield


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "asyncio_timeout(timeout): cancels the test execution after the specified "
        "number of seconds",
    )
    config.addinivalue_line(
        "markers",
        "rerun_on_failure(reruns): reruns test if it fails for the specified number "
        "of reruns",
    )
    config.addinivalue_line("markers", "slow: mark test as slow to run")
