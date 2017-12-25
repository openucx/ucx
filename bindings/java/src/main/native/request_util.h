/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef REQUEST_UTIL_H_
#define REQUEST_UTIL_H_

#include <ucs/type/status.h>
#include <ucp/api/ucp_def.h>

#include <cstdint>

/**
 * class to hold the request struct and request handlers
 */
class request_util {
public:
    struct request_t {
        // User defined request
        // Data members will be added later
    };

    class request_handler {
    public:
        static void request_init(void *request);

    private:
        request_handler() {}
    };

private:
    request_util() {}
};

typedef request_util::request_t jucx_request;

#endif /* REQUEST_UTIL_H_ */
