/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef SRC_WORKER_H_
#define SRC_WORKER_H_

#include "context.h"
#include "request_util.h"
#include "completion_queue.h"

#include <ucp/api/ucp.h>

#include <memory>
#include <atomic>

class worker {
public:
    worker(context* ctx, uint32_t cap, ucp_worker_params_t params);

    ~worker();

    ucs_status_t extract_worker_address(ucp_address_t** worker_address,
                                        size_t& address_length);

    void release_worker_address(ucp_address_t* worker_address);

    char *get_event_queue() const {
        return comp_queue->get_event_queue();
    }

    uint32_t get_queue_size() {
        return comp_queue->get_queue_size();
    }

    ucp_worker_h get_ucp_worker() const {
        return ucp_worker;
    }

    void put_in_event_queue(const command& cmd);

    void add_to_request_list(jucx_request* request);

    int progress();

private:
    context*        jucx_context;
    ucp_worker_h    ucp_worker;
    std::atomic_int event_cnt;
    jucx_request*   head;
    std::unique_ptr<completion_queue> comp_queue;

    void free_requests();
};

using native_ptr = uintptr_t;

#endif /* SRC_WORKER_H_ */
