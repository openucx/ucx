# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import ucp_py as ucp
import time
import argparse
import asyncio
import concurrent.futures
import sys

accept_cb_started = False
new_client_ep = None
max_msg_log = 23
max_iters = 1000
ignore_args = None

def allocate_mem(size, args):

    if not args.use_obj:
        send_buffer_region = ucp.buffer_region()
        recv_buffer_region = ucp.buffer_region()

        if args.mem_type == 'cuda':
            send_buffer_region.alloc_cuda(size)
            if not args.blind_recv:
                recv_buffer_region.alloc_cuda(size)
        else:
            send_buffer_region.alloc_host(size)
            if not args.blind_recv:
                recv_buffer_region.alloc_host(size)
    else:
        if args.mem_type == 'cuda':
            # Hopefully some day
            print("cuda objects not supported yet")
            sys.exit() 
        else:
            send_buffer_region = str(list(range(size)))
            recv_buffer_region = str(list(range(size, 2 * size)))

    return send_buffer_region, recv_buffer_region

def free_mem(send_buffer_region, recv_buffer_region, args):

    if not args.use_obj:
        if args.mem_type == 'cuda':
            send_buffer_region.free_cuda()
            if not args.blind_recv:
                recv_buffer_region.free_cuda()
        else:
            send_buffer_region.free_host()
            if not args.blind_recv:
                recv_buffer_region.free_host()

def profile(fxn, *args):
    tmp_start = time.time()
    rval = None
    if None == ignore_args:
        rval = fxn(*args)
    else:
        if fxn in ignore_args:
            rval = fxn()
        else:
            rval = fxn(*args)
    tmp_end = time.time()
    return rval, (tmp_end - tmp_start)

async def async_profile(fxn, *args):
    tmp_start = time.time()
    rval = None
    if None == ignore_args:
        rval = await fxn(*args)
    else:
        if fxn in ignore_args:
            rval = await fxn()
        else:
            rval = await fxn(*args)
    tmp_end = time.time()
    return rval, (tmp_end - tmp_start)

def populate_ops(ep, send_first, args):
    global ignore_args
    first_op = None
    second_op = None

    if send_first:
        if args.use_fast:
            first_op = ep.send_fast
            second_op = ep.recv_fast
        elif args.use_obj:
            first_op = ep.send_obj
            second_op = ep.recv_obj
        else:
            first_op = ep.send
            second_op = ep.recv

        if args.blind_recv:
            second_op = ep.recv_future
            ignore_args = []
            ignore_args.append(ep.recv_future)
    else:
        if args.use_fast:
            first_op = ep.recv_fast
            second_op = ep.send_fast
        elif args.use_obj:
            first_op = ep.recv_obj
            second_op = ep.send_obj
        else:
            first_op = ep.recv
            second_op = ep.send

        if args.blind_recv:
            first_op = ep.recv_future
            ignore_args = []
            ignore_args.append(ep.recv_future)

    return first_op, second_op

def run_iters(ep, first_buffer_region, second_buffer_region, msg_log, send_first, args):
    print("{}\t{}\t{}\t{}".format("Size (bytes)", "Latency (us)", "BW (GB/s)",
                                        "Issue (us)", "Progress (us)"))

    warmup_iters = int(0.1 * max_iters)

    first_op, second_op = populate_ops(ep, send_first, args)

    for i in range(msg_log):
        msg_len = 2 ** i

        first_msg = []
        second_msg = []
        for j in range(max_iters + warmup_iters):
            if not args.use_obj:
                first_msg.append(ucp.ucp_msg(first_buffer_region))
                second_msg.append(ucp.ucp_msg(second_buffer_region))
            else:
                first_msg.append(first_buffer_region)
                second_msg.append(second_buffer_region)

        start = time.time()
        issue_lat = 0
        progress_lat = 0
        
        if args.use_obj:
            msg_len = sys.getsizeof(str(list(range(msg_len))))

        for j in range(max_iters):

            first_req, time_spent = profile(first_op, first_msg[j], msg_len)
            if j >= warmup_iters:
                issue_lat += time_spent

            rval, time_spent = profile(first_req.result)
            if j >= warmup_iters:
                progress_lat += time_spent

            second_req, time_spent = profile(second_op, second_msg[j], msg_len)
            if j >= warmup_iters:
                issue_lat += time_spent

            rval, time_spent = profile(second_req.result)
            if j >= warmup_iters:
                progress_lat += time_spent

        end = time.time()
        lat = end - start
        get_avg_us = lambda x: ((x/2) / max_iters) * 1000000
        print("{}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}".format(msg_len, get_avg_us(lat),
                                                                  (((msg_len * max_iters)/(lat/2)) / 10 ** 9),
                                                                  get_avg_us(issue_lat),
                                                                  get_avg_us(progress_lat)))

async def run_iters_async(ep, first_buffer_region, second_buffer_region, msg_log, send_first, args):
    print("{}\t{}\t{}\t{}".format("Size (bytes)", "Latency (us)", "BW (GB/s)",
                                        "Issue (us)", "Progress (us)"))

    warmup_iters = int(0.1 * max_iters)

    first_op, second_op = populate_ops(ep, send_first, args)

    for i in range(msg_log):
        msg_len = 2 ** i

        if not args.use_obj:
            first_msg = ucp.ucp_msg(first_buffer_region)
            second_msg = ucp.ucp_msg(second_buffer_region)
        else:
            first_msg = first_buffer_region
            second_msg = second_buffer_region

        warmup_iters = int((0.1 * max_iters))
        if args.use_obj:
            msg_len = sys.getsizeof(str(list(range(msg_len))))
            
        for j in range(warmup_iters):
            first_req = await first_op(first_msg, msg_len)
            second_req = await second_op(second_msg, msg_len)

        start = time.time()
        for j in range(max_iters):
            send_req = await first_op(first_msg, msg_len)
            recv_req = await second_op(second_msg, msg_len)
        end = time.time()
        lat = end - start
        get_avg_us = lambda x: ((x/2) / max_iters) * 1000000
        print("{}\t\t{:.2f}\t\t{:.2f}".format(msg_len, get_avg_us(lat),
                                              (((msg_len * max_iters)/(lat/2)) / 10 ** 9)))

def talk_to_client(ep):

    global args
    global cb_not_done
    send_first = True

    send_buffer_region, recv_buffer_region = allocate_mem((1 << max_msg_log), args)
    run_iters(ep, send_buffer_region, recv_buffer_region, max_msg_log, send_first, args)
    free_mem(send_buffer_region, recv_buffer_region, args)

    ucp.destroy_ep(ep)
    cb_not_done = False
    ucp.stop_listener()

def talk_to_server(ip, port):

    global args

    ep = ucp.get_endpoint(ip, port)
    send_first = False

    send_buffer_region, recv_buffer_region = allocate_mem((1 << max_msg_log), args)
    run_iters(ep, recv_buffer_region, send_buffer_region, max_msg_log, send_first, args)
    free_mem(send_buffer_region, recv_buffer_region, args)

    ucp.destroy_ep(ep)

async def talk_to_client_async(ep):

    global args
    send_first = True

    send_buffer_region, recv_buffer_region = allocate_mem((1 << max_msg_log), args)
    await run_iters_async(ep, send_buffer_region, recv_buffer_region, max_msg_log, send_first, args)
    free_mem(send_buffer_region, recv_buffer_region, args)

    ucp.destroy_ep(ep)
    ucp.stop_listener()

async def talk_to_server_async(ip, port):

    global args

    ep = ucp.get_endpoint(ip, port)
    send_first = False

    send_buffer_region, recv_buffer_region = allocate_mem((1 << max_msg_log), args)
    await run_iters_async(ep, recv_buffer_region, send_buffer_region, max_msg_log, send_first, args)
    free_mem(send_buffer_region, recv_buffer_region, args)

    ucp.destroy_ep(ep)

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
parser.add_argument('-i','--intra_node', action='store_true')
parser.add_argument('-m','--mem_type', help='host/cuda (default = host)', required=False)
parser.add_argument('-a','--use_asyncio', help='use asyncio execution (default = false)', action="store_true")
parser.add_argument('-f','--use_fast', help='use fast send/recv (default = false)', action="store_true")
parser.add_argument('-o','--use_obj', help='use python objects for send/recv (default = false)', action="store_true")
parser.add_argument('-b','--blind_recv', help='use blind recv (default = false)', action="store_true")
parser.add_argument('-w','--wait', help='wait after every send/recv (default = false)', action="store_true")
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False
cb_not_done = True
if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

ucp.init()

if not args.use_asyncio:
    if server:
        if args.intra_node:
            ucp.set_cuda_dev(1)
        ucp.start_listener(talk_to_client, is_coroutine = False)
        while cb_not_done:
            ucp.progress()
    else:
        talk_to_server(init_str.encode(), int(args.port))
else:
    loop = asyncio.get_event_loop()
    if server:
        if args.intra_node:
            ucp.set_cuda_dev(1)
        coro = ucp.start_listener(talk_to_client_async, is_coroutine = True)
    else:
        coro = talk_to_server_async(init_str.encode(), int(args.port))

    loop.run_until_complete(coro)
    loop.close()

ucp.fin()
