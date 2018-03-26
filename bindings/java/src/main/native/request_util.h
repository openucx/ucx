/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef REQUEST_UTIL_H_
#define REQUEST_UTIL_H_

#include "commands.h"

#include <ucs/type/status.h>
#include <ucp/api/ucp_def.h>

#include <cstdint>

// forward declaration
class worker;

/**
 * class to hold the request struct and request handlers
 */
class request_util {
public:
    struct request_t {
        worker*     request_worker;
        ucp_ep_h    request_endpoint;
        command     cmd;
        int         done;
        request_t*  next;
    };

    class request_handler {
    public:
        static void request_init(void *request);

        static void recv_completion_handler(void* request, ucs_status_t status,
                                            size_t length);

        static void send_completion_handler(void* request, ucs_status_t status);

    private:
        static void stream_request_handler(request_t* request,
                                           ucs_status_t status);

        request_handler() {}
    };

    static int check_stream_request(request_t* request,
                                    worker* request_worker,
                                    ucp_ep_h request_endpoint,
                                    uint64_t request_id,
                                    CommandType type,
                                    size_t length = 0);

private:
    static CompletionStatus translate_status(ucs_status_t ucs_status);

    request_util() {}
};

using jucx_request = request_util::request_t;
using jucx_handler = request_util::request_handler;

#endif /* REQUEST_UTIL_H_ */
