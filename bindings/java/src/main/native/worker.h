/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef SRC_WORKER_H_
#define SRC_WORKER_H_

#include "context.h"

#include <ucp/api/ucp.h>

#include <cstddef>

class worker {
public:
    worker(context* ctx, uint32_t cap, ucp_worker_params_t params);

    ~worker();

    ucs_status_t extract_worker_address(ucp_address_t** worker_address,
                                        size_t& address_length);

    void release_worker_address(ucp_address_t* worker_address);

    char *get_event_queue() const {
        return event_queue;
    }

    uint32_t get_queue_size() const {
        return queue_size;
    }

    ucp_worker_h get_ucp_worker() const {
        return ucp_worker;
    }

private:
    context*        jucx_context;
    ucp_worker_h    ucp_worker;
    uint32_t        queue_size;
    char*           event_queue;
};

#endif /* SRC_WORKER_H_ */
