# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import logging
import struct

from ucxx._lib.arr import Array

from .utils import hash64bits

logger = logging.getLogger("ucx")


async def exchange_peer_info(endpoint, msg_tag, listener, connect_timeout=5.0):
    """Help function that exchange endpoint information"""

    # Pack peer information incl. a checksum
    fmt = "QQ"
    my_info = struct.pack(fmt, msg_tag, hash64bits(msg_tag))
    peer_info = bytearray(len(my_info))
    my_info_arr = Array(my_info)
    peer_info_arr = Array(peer_info)

    # Send/recv peer information. Notice, we force an `await` between the two
    # streaming calls (see <https://github.com/rapidsai/ucx-py/pull/509>)
    if listener is True:
        req = endpoint.stream_send(my_info_arr)
        await asyncio.wait_for(req.wait(), timeout=connect_timeout)
        req = endpoint.stream_recv(peer_info_arr)
        await asyncio.wait_for(req.wait(), timeout=connect_timeout)
    else:
        req = endpoint.stream_recv(peer_info_arr)
        await asyncio.wait_for(req.wait(), timeout=connect_timeout)
        req = endpoint.stream_send(my_info_arr)
        await asyncio.wait_for(req.wait(), timeout=connect_timeout)

    # Unpacking and sanity check of the peer information
    ret = {}
    (ret["msg_tag"], ret["checksum"]) = struct.unpack(fmt, peer_info)

    expected_checksum = hash64bits(ret["msg_tag"])

    if expected_checksum != ret["checksum"]:
        raise RuntimeError(
            f"Checksum invalid! {hex(expected_checksum)} != {hex(ret['checksum'])}"
        )

    return ret
