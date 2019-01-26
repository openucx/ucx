# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
#
# Description: Check if a process can function as both a listening
# server and as a client that connects to a server
#
# Steps:
# 1. Process starts as a server, activates listener
#    a. Obtains the coroutine that accepts incoming connection
#       -> coro_server
# 2. sleeps for 10s (waiting for the other server to come up)
# 3. Obtains the coroutine that connects to other server
#    -> coro_client
# 4. The process runs both (coro_server, coro_client) to completion


import ucp_py as ucp
import time
import argparse
import asyncio
import concurrent.futures

max_msg_log = 23

async def talk_to_client(ep):

    global max_msg_log
    global args

    start_string = "in talk_to_client"
    if args.blind_recv:
        start_string += " + blind recv"
    if args.check_data:
        start_string += " + data validity check"
    print(start_string)
    msg_log = max_msg_log

    send_buffer_region = ucp.buffer_region()
    send_buffer_region.alloc_cuda(1 << msg_log)

    send_msg = ucp.ucp_msg(send_buffer_region)

    recv_msg = None
    recv_buffer_region = None
    recv_req = None

    if not args.blind_recv:
        recv_buffer_region = ucp.buffer_region()
        recv_buffer_region.alloc_cuda(1 << msg_log)
        recv_msg = ucp.ucp_msg(recv_buffer_region)

    if args.check_data:
        send_msg.set_mem(0, 1 << msg_log)
        if not args.blind_recv:
            recv_msg.set_mem(0, 1 << msg_log)

    send_req = await ep.send(send_msg, 1 << msg_log)

    if not args.blind_recv:
        recv_req = await ep.recv(recv_msg, 1 << msg_log)
    else:
        recv_req = await ep.recv_future()

    if args.check_data:
        errs = 0
        errs = recv_req.check_mem(1, 1 << msg_log)
        print("num errs: " + str(errs))

    send_buffer_region.free_cuda()
    if not args.blind_recv:
        recv_buffer_region.free_cuda()

    ucp.destroy_ep(ep)
    print("done with talk_to_client")
    ucp.stop_listener()

async def talk_to_server(ip, port):

    global max_msg_log
    global args

    start_string = "in talk_to_server"
    if args.blind_recv:
        start_string += " + blind recv"
    if args.check_data:
        start_string += " + data validity check"
    print(start_string)

    msg_log = max_msg_log

    ep = ucp.get_endpoint(ip, port)

    send_buffer_region = ucp.buffer_region()
    send_buffer_region.alloc_cuda(1 << msg_log)

    send_msg = ucp.ucp_msg(send_buffer_region)

    recv_msg = None
    recv_buffer_region = None
    recv_req = None

    if not args.blind_recv:
        recv_buffer_region = ucp.buffer_region()
        recv_buffer_region.alloc_cuda(1 << msg_log)
        recv_msg = ucp.ucp_msg(recv_buffer_region)

    if args.check_data:
        send_msg.set_mem(1, 1 << msg_log)
        if not args.blind_recv:
            recv_msg.set_mem(1, 1 << msg_log)

    if not args.blind_recv:
        recv_req = await ep.recv(recv_msg, 1 << msg_log)
    else:
        recv_req = await ep.recv_future()

    send_req = await ep.send(send_msg, 1 << msg_log)

    if args.check_data:
        errs = 0
        errs = recv_req.check_mem(0, 1 << msg_log)
        print("num errs: " + str(errs))

    send_buffer_region.free_cuda()
    if not args.blind_recv:
        recv_buffer_region.free_cuda()

    ucp.destroy_ep(ep)
    print("done with talk_to_server")

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
parser.add_argument('-m','--my_port', help='enter own port number', required=False)
parser.add_argument('-c','--check_data', help='Check if data is valid. Default = False', action="store_true")
parser.add_argument('-b','--blind_recv', help='Use blind recv. Default = False', action="store_true")
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False
if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

ucp.init()
loop = asyncio.get_event_loop()
# coro points to either client or server-side coroutine
coro_server = ucp.start_listener(talk_to_client,
                                 listener_port = int(args.my_port),
                                 is_coroutine = True)
time.sleep(5)
coro_client = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(
    asyncio.gather(coro_server, coro_client)
)

loop.close()
ucp.fin()
