# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import pickle

import ucxx._lib.libucxx as ucx_api

mp = mp.get_context("spawn")


def test_ucx_address():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    new_address = ucx_api.UCXAddress.create_from_buffer(org_address)
    assert hash(org_address) == hash(new_address)
    assert bytes(org_address) == bytes(new_address)


def test_ucx_address_bytes():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_bytes = bytes(org_address)
    new_address = ucx_api.UCXAddress.create_from_buffer(org_address_bytes)
    new_address_bytes = bytes(new_address)
    assert hash(org_address) == hash(new_address)
    assert org_address_bytes == new_address_bytes


def test_ucx_address_bytearray():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_bytearray = bytearray(org_address)
    new_address = ucx_api.UCXAddress.create_from_buffer(org_address_bytearray)
    new_address_bytearray = bytearray(new_address)
    assert hash(org_address) == hash(new_address)
    assert org_address_bytearray == new_address_bytearray


def test_ucx_address_memoryview():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_mv = memoryview(org_address)
    new_address = ucx_api.UCXAddress.create_from_buffer(org_address_mv)
    new_address_mv = memoryview(new_address)
    assert hash(org_address) == hash(new_address)
    assert org_address_mv == new_address_mv


def test_ucx_address_picklebuffer():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_pb = pickle.PickleBuffer(org_address)
    new_address = ucx_api.UCXAddress.create_from_buffer(org_address_pb)
    new_address_pb = pickle.PickleBuffer(new_address)
    assert hash(org_address) == hash(new_address)
    assert org_address_pb.raw() == new_address_pb.raw()


def test_pickle_ucx_address():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_bytes = bytes(org_address)
    org_address_hash = hash(org_address)
    dumped_address = pickle.dumps(org_address)
    new_address = pickle.loads(dumped_address)
    new_address_bytes = bytes(new_address)

    assert org_address_hash == hash(new_address)
    assert org_address_bytes == new_address_bytes


def test_pickle5_ucx_address():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_bytes = bytes(org_address)
    org_address_hash = hash(org_address)
    dumped_address_list = [None]
    dumped_address_list[0] = pickle.dumps(
        org_address, protocol=5, buffer_callback=dumped_address_list.append
    )
    assert len(dumped_address_list) == 2
    assert isinstance(dumped_address_list[1], pickle.PickleBuffer)
    assert dumped_address_list[1].raw().obj is org_address

    new_address = pickle.loads(dumped_address_list[0], buffers=dumped_address_list[1:])
    new_address_bytes = bytes(new_address)
    assert org_address_hash == hash(new_address)
    assert org_address_bytes == new_address_bytes
