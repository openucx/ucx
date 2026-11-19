# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ucxx._lib.libucxx as ucx_api


def _init_and_get_objects(progress_mode):
    """Initialize and return UCXX objects

    Initialize a UCXX worker and listener/client pair returning the objects.
    """
    feature_flags = [ucx_api.Feature.WAKEUP, ucx_api.Feature.TAG]

    ctx = ucx_api.UCXContext(feature_flags=tuple(feature_flags))
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early and allow transfers outside of the listsner's
    # callback even after it has terminated.
    listener_ep = [None]

    def _listener_handler(ep):
        listener_ep[0] = ep

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler, deliver_endpoint=True
    )

    client_ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        listener.port,
        endpoint_error_handling=True,
    )

    while listener_ep[0] is None:
        if progress_mode == "blocking":
            worker.progress()

    if progress_mode == "thread":
        worker.stop_progress_thread()

    return (worker, client_ep, listener_ep[0])


@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_ucp_handles(progress_mode):
    """Test UCP handles.

    Test that UCP worker and endpoint handles are valid.
    """
    worker, client_ep, listener_ep = _init_and_get_objects(progress_mode)

    assert all([ep.worker_handle == worker.handle for ep in [client_ep, listener_ep]])
    assert all([isinstance(ep.handle, int) for ep in [client_ep, listener_ep]])
    assert all([ep.handle > 0 for ep in [client_ep, listener_ep]])


@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_ucxx_handles(progress_mode):
    """Test UCXX handles.

    Test that UCXX worker and endpoint handles are valid.
    """
    worker, client_ep, listener_ep = _init_and_get_objects(progress_mode)

    assert all(
        [ep.ucxx_worker_ptr == worker.ucxx_ptr for ep in [client_ep, listener_ep]]
    )
    assert all([isinstance(ep.ucxx_ptr, int) for ep in [client_ep, listener_ep]])
    assert all([ep.ucxx_ptr > 0 for ep in [client_ep, listener_ep]])
